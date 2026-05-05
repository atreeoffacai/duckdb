#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression_binder/lateral_binder.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_positional_join.hpp"
#include "duckdb/planner/subquery/recursive_dependent_join_planner.hpp"
#include "duckdb/planner/tableref/bound_joinref.hpp"

namespace duckdb {

//! Check if a filter can be safely pushed to the left child
//! This is used ONLY for join conditions in the ON clause, not for WHERE clause filters.
//! The logic determines whether a condition that references only the left side can be
//! pushed down as a filter on the left child operator.
static bool CanPushToLeftChild(JoinType type, JoinRefType ref_type) {
	// Unsupported arbitrary predicates for some ASOF types
	if (ref_type == JoinRefType::ASOF && type != JoinType::INNER && type != JoinType::LEFT) {
		return false;
	}

	switch (type) {
	case JoinType::INNER:
	case JoinType::SEMI:
	case JoinType::RIGHT:
		return true;
	case JoinType::ANTI:
	case JoinType::LEFT:
	case JoinType::OUTER:
		return false;
	default:
		return false;
	}
}

//! Check if a filter can be safely pushed to the right child
//! This is used ONLY for join conditions in the ON clause, not for WHERE clause filters.
//! The logic determines whether a condition that references only the right side can be
//! pushed down as a filter on the right child operator.
static bool CanPushToRightChild(JoinType type, JoinRefType ref_type) {
	// Unsupported arbitrary predicates for some ASOF types
	if (ref_type == JoinRefType::ASOF && type != JoinType::INNER && type != JoinType::LEFT) {
		return false;
	}

	switch (type) {
	case JoinType::INNER:
	case JoinType::SEMI:
	case JoinType::ANTI:
	case JoinType::LEFT:
		return true;
	case JoinType::RIGHT:
	case JoinType::OUTER:
		return false;
	default:
		return false;
	}
}

//! Push a filter expression to a child operator
static void PushFilterToChild(unique_ptr<LogicalOperator> &child, unique_ptr<Expression> &expr) {
	if (child->type != LogicalOperatorType::LOGICAL_FILTER) {
		auto filter = make_uniq<LogicalFilter>();
		filter->AddChild(std::move(child));
		child = std::move(filter);
	}

	auto &filter = child->Cast<LogicalFilter>();
	filter.expressions.push_back(std::move(expr));
}

//! Check if a foldable expression evaluates to TRUE and can be eliminated
static bool CanEliminate(ClientContext &context, JoinType type, unique_ptr<Expression> &expr) {
	if (!expr->IsFoldable()) {
		return false;
	}

	Value result;
	if (!ExpressionExecutor::TryEvaluateScalar(context, *expr, result)) {
		return false;
	}

	if (result.IsNull()) {
		return false;
	}

	bool is_true = (result == Value(true));

	if (is_true) {
		switch (type) {
		case JoinType::INNER:
		case JoinType::LEFT:
		case JoinType::RIGHT:
		case JoinType::SEMI:
		case JoinType::ANTI:
		case JoinType::OUTER:
			return true;
		default:
			return false;
		}
	}

	return false;
}

//! Only use conditions that are valid for the join ref type
static bool IsJoinTypeCondition(const JoinRefType ref_type, const ExpressionType expr_type) {
	switch (ref_type) {
	case JoinRefType::ASOF:
		switch (expr_type) {
		case ExpressionType::COMPARE_EQUAL: // =
		case ExpressionType::COMPARE_NOT_DISTINCT_FROM: // Not Distinct From
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO: // >=
		case ExpressionType::COMPARE_GREATERTHAN: // >
		case ExpressionType::COMPARE_LESSTHANOREQUALTO: // <=
		case ExpressionType::COMPARE_LESSTHAN: // <
			return true;
		default:
			return false;
		}
	default:
		return true;
	}
}

//! 检查一个表达式是否为可用的比较表达式
static bool IsComparisonExpression(const Expression &expr) {
	switch (expr.GetExpressionType()) {
	case ExpressionType::COMPARE_EQUAL: // = 
	case ExpressionType::COMPARE_NOTEQUAL: // !=
	case ExpressionType::COMPARE_LESSTHAN: // <
	case ExpressionType::COMPARE_GREATERTHAN: // >
	case ExpressionType::COMPARE_LESSTHANOREQUALTO: // <=
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO: // >=
	case ExpressionType::COMPARE_NOT_DISTINCT_FROM: // Not Distinct From
	case ExpressionType::COMPARE_DISTINCT_FROM: // Distinct From
		return true;
	default:
		return false;
	}
}

//! Create a JoinCondition from a comparison
static bool CreateJoinCondition(Expression &expr, const unordered_set<TableIndex> &left_bindings,
                                const unordered_set<TableIndex> &right_bindings, vector<JoinCondition> &conditions) {
	// comparison
	auto &comparison = expr.Cast<BoundComparisonExpression>();
	auto left_side = JoinSide::GetJoinSide(*comparison.left, left_bindings, right_bindings);
	auto right_side = JoinSide::GetJoinSide(*comparison.right, left_bindings, right_bindings);
	if (left_side != JoinSide::BOTH && right_side != JoinSide::BOTH) {
		// join condition can be divided in a left/right side
		auto comp_type = expr.GetExpressionType();
		auto left = std::move(comparison.left);
		auto right = std::move(comparison.right);
		if (left_side == JoinSide::RIGHT) {
			// left = right, right = left, flip the comparison symbol and reverse sides
			swap(left, right);
			comp_type = FlipComparisonExpression(comp_type);
		}
		conditions.push_back(JoinCondition(std::move(left), std::move(right), comp_type));
		return true;
	}
	return false;
}

//! 提取连接条件，在安全的情况下将单侧过滤器下推到子节点
void LogicalComparisonJoin::ExtractJoinConditions(ClientContext &context, JoinType type, JoinRefType ref_type,
                                                  unique_ptr<LogicalOperator> &left_child,
                                                  unique_ptr<LogicalOperator> &right_child,
                                                  const unordered_set<TableIndex> &left_bindings,
                                                  const unordered_set<TableIndex> &right_bindings,
                                                  vector<unique_ptr<Expression>> &expressions,
                                                  vector<JoinCondition> &conditions) {
	for (auto &expr : expressions) {
		auto side = JoinSide::GetJoinSide(*expr, left_bindings, right_bindings);

		if (side == JoinSide::NONE) { // 常量表达式，尽量消除它
			if (CanEliminate(context, type, expr)) {
				continue;
			}
		} else if (side == JoinSide::LEFT) { // 单测过滤条件，往单测处下推
			if (CanPushToLeftChild(type, ref_type)) {
				PushFilterToChild(left_child, expr);
				continue;
			}
		} else if (side == JoinSide::RIGHT) { // 单测过滤条件，往单测处下推
			if (CanPushToRightChild(type, ref_type)) {
				PushFilterToChild(right_child, expr);
				continue;
			}
		} else if (side == JoinSide::BOTH) { // 双边引用条件
			if (IsComparisonExpression(*expr) && IsJoinTypeCondition(ref_type, expr->GetExpressionType()) &&
			    CreateJoinCondition(*expr, left_bindings, right_bindings, conditions)) {
				continue;
			}
		}

		conditions.emplace_back(std::move(expr)); // 这种JoinCondition就是INVALID，是一个奇怪的条件
	}
}

void LogicalComparisonJoin::ExtractJoinConditions(ClientContext &context, JoinType type, JoinRefType ref_type,
                                                  unique_ptr<LogicalOperator> &left_child,
                                                  unique_ptr<LogicalOperator> &right_child,
                                                  vector<unique_ptr<Expression>> &expressions,
                                                  vector<JoinCondition> &conditions) {
	unordered_set<TableIndex> left_bindings, right_bindings;
	LogicalJoin::GetTableReferences(*left_child, left_bindings);
	LogicalJoin::GetTableReferences(*right_child, right_bindings);
	return ExtractJoinConditions(context, type, ref_type, left_child, right_child, left_bindings, right_bindings,
	                             expressions, conditions);
}

void LogicalComparisonJoin::ExtractJoinConditions(ClientContext &context, JoinType type, JoinRefType ref_type,
                                                  unique_ptr<LogicalOperator> &left_child,
                                                  unique_ptr<LogicalOperator> &right_child,
                                                  unique_ptr<Expression> condition, vector<JoinCondition> &conditions) {
	// split the expressions by the AND clause
	vector<unique_ptr<Expression>> expressions;
	expressions.push_back(std::move(condition));
	LogicalFilter::SplitPredicates(expressions);
	return ExtractJoinConditions(context, type, ref_type, left_child, right_child, expressions, conditions);
}

//! 根据条件和连接类型创建连接操作符
// JoinType (连接类型)：决定了结果集长什么样，即如何处理匹配和不匹配的行。
// 比如 INNER、LEFT、RIGHT、FULL OUTER、SEMI、ANTI 等。它是经典关系代数中的概念。
// JoinRefType (连接引用类型/语义类型)：决定了这个连接的特殊业务语义或语法来源。
// 它告诉绑定器（Binder）这个 Join 是不是某种带有特殊匹配规则的变体。
unique_ptr<LogicalOperator> LogicalComparisonJoin::CreateJoin(JoinType type, JoinRefType ref_type,
                                                              unique_ptr<LogicalOperator> left_child,
                                                              unique_ptr<LogicalOperator> right_child,
                                                              vector<JoinCondition> conditions) {
	// 为验证目的，分离比较条件和非比较条件： separate comparison and non-comparison conditions for validation
	vector<JoinCondition> comparison_conditions;
	vector<JoinCondition> non_comparison_conditions;
	for (auto &cond : conditions) {
		if (cond.IsComparison()) {
			comparison_conditions.push_back(std::move(cond));
		} else {
			non_comparison_conditions.push_back(std::move(cond));
		}
	}

	// 取消asof，如果不取消，验证asof是否合法
	auto is_asof = (ref_type == JoinRefType::ASOF);
	if (is_asof) {
		switch (type) {
		case JoinType::RIGHT:
		case JoinType::OUTER:
			// 对于某些ASOF连接，我们（目前）还不能支持任意的谓词条件
			if (!non_comparison_conditions.empty()) {
				throw NotImplementedException("Unsupported ASOF JOIN type (%s) with arbitrary predicate",
				                              EnumUtil::ToChars(type));
			}
			break;
		case JoinType::SEMI:
		case JoinType::ANTI:
			// 对于这些连接类型，我们可以使用普通连接，因为右侧匹配并不重要(asof 语义失效，没必要保持asof)
			// 但我们会验证ASOF的要求
			is_asof = false;
			ref_type = JoinRefType::REGULAR;
			break;
		default:
			break;
		}
		// xm: 检查asof join,确保只有一个非等值比较条件，并且它是一个有效的asof比较（等值比较可以有很多）
		idx_t asof_idx = comparison_conditions.size();
		for (size_t c = 0; c < comparison_conditions.size(); ++c) {
			auto &cond = comparison_conditions[c];
			switch (cond.GetComparisonType()) {
			case ExpressionType::COMPARE_EQUAL: // =
			case ExpressionType::COMPARE_NOT_DISTINCT_FROM: // Not Distinct From
				break; // xm: switch 内部的 break 只会跳出当前的 switch 块，不会终止循环
			case ExpressionType::COMPARE_GREATERTHANOREQUALTO: // >=
			case ExpressionType::COMPARE_GREATERTHAN: // >
			case ExpressionType::COMPARE_LESSTHANOREQUALTO: // <=
			case ExpressionType::COMPARE_LESSTHAN: // <
				if (asof_idx < comparison_conditions.size()) {
					throw BinderException("Multiple ASOF JOIN inequalities");
				}
				asof_idx = c;
				break;
			default:
				throw BinderException("Invalid ASOF JOIN comparison");
			}
		}
		if (asof_idx >= comparison_conditions.size()) {
			throw BinderException("Missing ASOF JOIN inequality");
		}
	}

	// 重构完整的条件向量（vector）,这里涉及到C++的移动语义（move semantics）：
	// comparison_conditions中的元素变了（被掏空了），但是其元素个数并没有变，也就是size()的结果没有变
	vector<JoinCondition> all_conditions;
	for (auto &cond : comparison_conditions) {
		all_conditions.push_back(std::move(cond));
	}
	for (auto &cond : non_comparison_conditions) {
		all_conditions.push_back(std::move(cond));
	}

	// 准备工作已经就绪，现在根据不同的情况创建连接操作符：
	// 情况1：ASOF连接 - 使用比较连接（所有条件已经在向量中）
	if (is_asof) {
		auto asof_join = make_uniq<LogicalComparisonJoin>(type, LogicalOperatorType::LOGICAL_ASOF_JOIN);
		asof_join->conditions = std::move(all_conditions);
		asof_join->children.push_back(std::move(left_child));
		asof_join->children.push_back(std::move(right_child));
		return std::move(asof_join);
	}

	// 情况2：没有比较条件 - 使用any join（此时可能存在非比较条件）
	if (comparison_conditions.empty()) {
		if (all_conditions.empty()) { // 没有任何条件，包括比较条件和非比较快条件
			all_conditions.emplace_back(make_uniq<BoundConstantExpression>(Value::BOOLEAN(true)));
		}

		auto any_join = make_uniq<LogicalAnyJoin>(type);
		any_join->children.push_back(std::move(left_child));
		any_join->children.push_back(std::move(right_child));
		any_join->condition = JoinCondition::CreateExpression(std::move(all_conditions));
		return std::move(any_join);
	}

	// 情况3：有比较条件 - 使用比较连接
	auto comp_join = make_uniq<LogicalComparisonJoin>(type, LogicalOperatorType::LOGICAL_COMPARISON_JOIN);
	comp_join->conditions = std::move(all_conditions);
	comp_join->children.push_back(std::move(left_child));
	comp_join->children.push_back(std::move(right_child));

	return std::move(comp_join);
}

static bool HasCorrelatedColumns(const Expression &root_expr) {
	bool has_correlated_columns = false;
	ExpressionIterator::VisitExpression<BoundColumnRefExpression>(root_expr,
	                                                              [&](const BoundColumnRefExpression &colref) {
		                                                              if (colref.depth > 0) {
			                                                              has_correlated_columns = true;
		                                                              }
	                                                              });
	return has_correlated_columns;
}

unique_ptr<LogicalOperator> LogicalComparisonJoin::CreateJoin(ClientContext &context, JoinType type,
                                                              JoinRefType reftype,
                                                              unique_ptr<LogicalOperator> left_child,
                                                              unique_ptr<LogicalOperator> right_child,
                                                              unique_ptr<Expression> condition) {
	vector<JoinCondition> conditions;
	LogicalComparisonJoin::ExtractJoinConditions(context, type, reftype, left_child, right_child, std::move(condition),
	                                             conditions);
	return LogicalComparisonJoin::CreateJoin(type, reftype, std::move(left_child), std::move(right_child),
	                                         std::move(conditions));
}

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundJoinRef &ref) {
	auto old_is_outside_flattened = is_outside_flattened;
	// 从最外层到最内层规划横向连接（LATERAL）,Plan laterals from outermost to innermost
	if (ref.lateral) {
		// Set the flag to ensure that children do not flatten before the root,xm:// 设置标志位，确保子节点在根节点之前不会被扁平化
		is_outside_flattened = false;
	}
	// xm: 准备左子树和右子树
	auto left = std::move(ref.left.plan);
	auto right = std::move(ref.right.plan);
	is_outside_flattened = old_is_outside_flattened;

	// 对于连接操作，由于横向绑定器（lateral binder）的存在，右侧绑定的深度会比左侧高1
	// 如果当前连接在左右两侧之间没有关联关系，那么右侧绑定的深度会整体偏高1，可以在整个过程中将其减少1
	if (!ref.lateral && !ref.correlated_columns.empty()) {
		LateralBinder::ReduceExpressionDepth(*right, ref.correlated_columns);
	}
	// xm: 将右外连接转换为左外连接
	if (ref.type == JoinType::RIGHT && ref.ref_type != JoinRefType::ASOF &&
	    ClientConfig::GetConfig(context).enable_optimizer &&
	    !Optimizer::OptimizerDisabled(context, OptimizerType::BUILD_SIDE_PROBE_SIDE)) {
		// 为了优化目的，我们将任何右外连接转换为左外连接
		// 它们本质上是相同的，只是左右两侧互换，因此将它们同等对待可以简化处理
		ref.type = JoinType::LEFT;
		std::swap(left, right);
	}
	if (ref.lateral) {
		auto new_plan = PlanLateralJoin(std::move(left), std::move(right), ref.correlated_columns, ref.type,
		                                std::move(ref.condition));
		if (has_unplanned_dependent_joins) {
			RecursiveDependentJoinPlanner plan(*this);
			plan.VisitOperator(*new_plan);
		}
		return new_plan;
	}
	// xm: 跳转处理特殊连接类型
	switch (ref.ref_type) {
	case JoinRefType::CROSS:
		return LogicalCrossProduct::Create(std::move(left), std::move(right));
	case JoinRefType::POSITIONAL:
		return LogicalPositionalJoin::Create(std::move(left), std::move(right));
	default:
		break;
	}
	// xm: 处理其他连接类型
	if (ref.type == JoinType::INNER && (ref.condition->HasSubquery() || HasCorrelatedColumns(*ref.condition)) &&
	    ref.ref_type == JoinRefType::REGULAR) {
		// 内连接，生成笛卡尔积 + 过滤条件
		// 之后将由连接顺序优化器将其转换为合适的连接操作
		auto root = LogicalCrossProduct::Create(std::move(left), std::move(right));

		auto filter = make_uniq<LogicalFilter>(std::move(ref.condition));
		// visit the expressions in the filter
		for (auto &expression : filter->expressions) {
			PlanSubqueries(expression, root);
		}
		filter->AddChild(std::move(root));
		return std::move(filter);
	}

	// now create the join operator from the join condition
	auto result = LogicalComparisonJoin::CreateJoin(context, ref.type, ref.ref_type, std::move(left), std::move(right),
	                                                std::move(ref.condition));
	optional_ptr<LogicalOperator> join;
	if (result->type == LogicalOperatorType::LOGICAL_FILTER) {
		join = result->children[0].get();
	} else {
		join = result.get();
	}

	if (ref.type == JoinType::MARK) {
		join->Cast<LogicalJoin>().mark_index = ref.mark_index;
	}
	for (auto &child : join->children) {
		if (child->type == LogicalOperatorType::LOGICAL_FILTER) {
			auto &filter = child->Cast<LogicalFilter>();
			for (auto &expr : filter.expressions) {
				PlanSubqueries(expr, filter.children[0]); // 处理表达式中的子查询
			}
		}
	}

	// we visit the expressions depending on the type of join
	switch (join->type) {
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		auto &comp_join = join->Cast<LogicalComparisonJoin>();
		for (idx_t i = 0; i < comp_join.conditions.size(); i++) {
			auto &cond = comp_join.conditions[i];
			if (cond.IsComparison()) {
				PlanSubqueries(cond.LeftReference(), comp_join.children[0]);  // 处理表达式中的子查询
				PlanSubqueries(cond.RightReference(), comp_join.children[1]);
			}
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_ANY_JOIN: {
		auto &any_join = join->Cast<LogicalAnyJoin>();
		// for the any join we just visit the condition
		if (any_join.condition->HasSubquery()) {
			throw NotImplementedException("Cannot perform non-inner join on subquery!");
		}
		break;
	}
	default:
		break;
	}
	if (!ref.duplicate_eliminated_columns.empty()) {
		auto &comp_join = join->Cast<LogicalComparisonJoin>();
		comp_join.type = LogicalOperatorType::LOGICAL_DELIM_JOIN;
		comp_join.delim_flipped = ref.delim_flipped;
		for (auto &col : ref.duplicate_eliminated_columns) {
			comp_join.duplicate_eliminated_columns.emplace_back(col->Copy());
		}
	}
	return result;
}

} // namespace duckdb
