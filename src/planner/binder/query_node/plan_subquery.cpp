#include "duckdb/function/aggregate/distributive_functions.hpp"
#include "duckdb/function/aggregate/distributive_function_utils.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_subquery_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/list.hpp"
#include "duckdb/planner/operator/logical_window.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/planner/subquery/flatten_dependent_join.hpp"
#include "duckdb/common/enums/logical_operator_type.hpp"
#include "duckdb/planner/operator/logical_dependent_join.hpp"
#include "duckdb/planner/subquery/recursive_dependent_join_planner.hpp"
#include "duckdb/function/scalar/generic_functions.hpp"
#include "duckdb/function/scalar/struct_functions.hpp"
#include "duckdb/main/settings.hpp"

namespace duckdb {

static unique_ptr<Expression> PlanUncorrelatedSubquery(Binder &binder, BoundSubqueryExpression &expr,
                                                       unique_ptr<LogicalOperator> &root,
                                                       unique_ptr<LogicalOperator> plan) {
	D_ASSERT(!expr.IsCorrelated());
	switch (expr.subquery_type) {
	case SubqueryType::EXISTS: {
		// 非关联 EXISTS
		// 我们只关心是否存在，因此我们推送一个 LIMIT 1 操作符
		auto limit = make_uniq<LogicalLimit>(BoundLimitNode::ConstantValue(1), BoundLimitNode());
		limit->AddChild(std::move(plan));
		plan = std::move(limit);

		// 现在我们在 LIMIT 上推送一个 COUNT(*) 聚合，结果将是 0 或 1（EXISTS 或 NOT EXISTS）
		auto count_star_fun = CountStarFun::GetFunction();

		FunctionBinder function_binder(binder);
		auto count_star =
		    function_binder.BindAggregateFunction(count_star_fun, {}, nullptr, AggregateType::NON_DISTINCT);
		auto idx_type = count_star->return_type;
		vector<unique_ptr<Expression>> aggregate_list;
		aggregate_list.push_back(std::move(count_star));
		auto aggregate_index = binder.GenerateTableIndex();
		auto aggregate =
		    make_uniq<LogicalAggregate>(binder.GenerateTableIndex(), aggregate_index, std::move(aggregate_list));
		aggregate->AddChild(std::move(plan));
		plan = std::move(aggregate);

		// 现在我们推送一个与 1 进行比较的投影
		auto left_child =
		    make_uniq<BoundColumnRefExpression>(idx_type, ColumnBinding(aggregate_index, ProjectionIndex(0)));
		auto right_child = make_uniq<BoundConstantExpression>(Value::Numeric(idx_type, 1));
		auto comparison = make_uniq<BoundComparisonExpression>(ExpressionType::COMPARE_EQUAL, std::move(left_child),
		                                                       std::move(right_child));

		vector<unique_ptr<Expression>> projection_list;
		projection_list.push_back(std::move(comparison));
		auto projection_index = binder.GenerateTableIndex();
		auto projection = make_uniq<LogicalProjection>(projection_index, std::move(projection_list));
		projection->AddChild(std::move(plan));
		plan = std::move(projection);

		// 我们通过添加一个笛卡尔积将其添加到主查询中
		// FIXME: 应该使用除笛卡尔积之外的其他方式，因为我们总是只添加一个标量常量
		root = LogicalCrossProduct::Create(std::move(root), std::move(plan));

		// 我们用一个 ColumnRefExpression 替换原始子查询，该表达式引用投影的结果（要么是 TRUE，要么是 FALSE）
		return make_uniq<BoundColumnRefExpression>(expr.GetName(), LogicalType::BOOLEAN,
		                                           ColumnBinding(projection_index, ProjectionIndex(0)));
	}
	case SubqueryType::SCALAR: {
		// 非关联标量查询，我们想要返回第一个条目
		// 找出我们想要返回的条目的绑定表的表索引
		auto bindings = plan->GetColumnBindings();
		D_ASSERT(bindings.size() == 1);
		auto table_idx = bindings[0].table_index;

		bool error_on_multiple_rows = Settings::Get<ScalarSubqueryErrorOnMultipleRowsSetting>(binder.context);

		// 我们推送一个返回FIRST元素的聚合操作
		vector<unique_ptr<Expression>> expressions;
		auto bound =
		    make_uniq<BoundColumnRefExpression>(expr.return_type, ColumnBinding(table_idx, ProjectionIndex(0)));
		vector<unique_ptr<Expression>> first_children;
		first_children.push_back(std::move(bound));

		FunctionBinder function_binder(binder);
		auto first_agg =
		    function_binder.BindAggregateFunction(FirstFunctionGetter::GetFunction(expr.return_type),
		                                          std::move(first_children), nullptr, AggregateType::NON_DISTINCT);

		expressions.push_back(std::move(first_agg));
		if (error_on_multiple_rows) {
			vector<unique_ptr<Expression>> count_children;
			auto count_agg = function_binder.BindAggregateFunction(
			    CountStarFun::GetFunction(), std::move(count_children), nullptr, AggregateType::NON_DISTINCT);
			expressions.push_back(std::move(count_agg));
		}
		auto aggr_index = binder.GenerateTableIndex();
		// xm：创建一个聚合函数
		auto aggr = make_uniq<LogicalAggregate>(binder.GenerateTableIndex(), aggr_index, std::move(expressions));
		aggr->AddChild(std::move(plan));
		plan = std::move(aggr);

		if (error_on_multiple_rows) {
			// CASE WHEN count > 1 THEN error('Scalar subquery can only return a single row') ELSE first_agg END
			auto proj_index = binder.GenerateTableIndex();

			auto first_ref = make_uniq<BoundColumnRefExpression>(plan->expressions[0]->return_type,
			                                                     ColumnBinding(aggr_index, ProjectionIndex(0)));
			auto count_ref = make_uniq<BoundColumnRefExpression>(plan->expressions[1]->return_type,
			                                                     ColumnBinding(aggr_index, ProjectionIndex(1)));

			auto constant_one = make_uniq<BoundConstantExpression>(Value::BIGINT(1));
			auto count_check = make_uniq<BoundComparisonExpression>(ExpressionType::COMPARE_GREATERTHAN,
			                                                        std::move(count_ref), std::move(constant_one));

			vector<unique_ptr<Expression>> error_children;
			error_children.push_back(make_uniq<BoundConstantExpression>(
			    Value("More than one row returned by a subquery used as an expression - scalar subqueries can only "
			          "return a single row.\n\nUse \"SET scalar_subquery_error_on_multiple_rows=false\" to revert to "
			          "previous behavior of returning a random row.")));
			auto error_expr = function_binder.BindScalarFunction(ErrorFun::GetFunction(), std::move(error_children));
			error_expr->return_type = first_ref->return_type;
			auto case_expr =
			    make_uniq<BoundCaseExpression>(std::move(count_check), std::move(error_expr), std::move(first_ref));

			vector<unique_ptr<Expression>> proj_expressions;
			proj_expressions.push_back(std::move(case_expr));

			auto proj = make_uniq<LogicalProjection>(proj_index, std::move(proj_expressions));
			proj->AddChild(std::move(plan));
			plan = std::move(proj);

			aggr_index = proj_index;
		}

		// 在非关联的情况下，我们通过笛卡尔积将值添加到主查询中
		// FIXME: 应该使用除笛卡尔积之外的其他方式，因为我们总是只添加一个标量常量，
		// 而笛卡尔积并没有针对这种情况进行优化。
		D_ASSERT(root);
		root = LogicalCrossProduct::Create(std::move(root), std::move(plan));

		// 我们将原始子查询替换为一个 BoundColumnRefExpression，
		// 该表达式引用聚合操作的第一个结果
		return make_uniq<BoundColumnRefExpression>(expr.GetName(), expr.return_type,
		                                           ColumnBinding(aggr_index, ProjectionIndex(0)));
	}
	default: {
		D_ASSERT(expr.subquery_type == SubqueryType::ANY);
		// 我们生成一个 MARK 连接，其结果为（TRUE、FALSE 或 NULL）
		// 子查询包含 NULL 值 -> 结果为（TRUE 或 NULL）
		// 子查询不包含 NULL 值 -> 结果为（TRUE、FALSE 或 NULL [如果输入为 NULL]）
		// 获取列绑定
		auto plan_columns = plan->GetColumnBindings();

		// then we generate the MARK join with the subquery
		auto mark_index = binder.GenerateTableIndex();
		auto join = make_uniq<LogicalComparisonJoin>(JoinType::MARK);
		join->mark_index = mark_index;
		join->AddChild(std::move(root));
		join->AddChild(std::move(plan));

		// 创建 JOIN 条件
		// 特殊情况：如果我们有一个单一的结构体子节点和多种类型，
		// 这意味着我们为了有序比较而保持了结构体的完整性（例如，(a,b) < ANY(...)）
		// 我们需要从子查询列中在右侧（RHS）构造一个对应的结构体
		if (expr.children.size() == 1 && expr.child_types.size() > 1) {
			// Construct a struct on the RHS from the subquery columns xm:其实就是基于子查询结果的列构造一个结构体表达式，以便进行“字段序比较”：
			// 在 SQL 语义中，(a, b) < (x, y) 是字典序比较（Lexicographical order）。它等价于 a < x OR (a = x AND b < y)。
			vector<unique_ptr<Expression>> struct_children;
			struct_children.reserve(expr.child_types.size());
			for (idx_t i = 0; i < expr.child_types.size(); i++) {
				auto &child_type = expr.child_types[i];
				auto &compare_type = expr.child_targets[i];
				auto colref = BoundCastExpression::AddDefaultCastToType(
				    make_uniq<BoundColumnRefExpression>(child_type, plan_columns[i]), compare_type);
				struct_children.push_back(std::move(colref));
			}

			// Create a struct expression from the subquery columns using the "row" function
			// xm: row() 函数是一个用于将多个独立的值或列“打包”成一个单一复合类型（通常称为 Struct、Tuple 或 Row 类型）的构造函数。
			// SELECT * FROM users WHERE (age, salary) > (25, 50000);
			// 在数据库的底层解析器看来，(25, 50000) 实际上会被转化并等价于：row(25, 50000)
			FunctionBinder function_binder(binder);
			auto struct_expr = function_binder.BindScalarFunction(RowFun::GetFunction(), std::move(struct_children));
			
			// xm: expr节点代表是any：例如：WHERE (a, b) < ANY (SELECT x, y FROM t)，expr.children[0]其实就是 (a, b) 
			JoinCondition cond(std::move(expr.children[0]), std::move(struct_expr), expr.comparison_type);

			// push collations
			ExpressionBinder::PushCollation(binder.context, cond.LeftReference(), cond.GetLHS().return_type);
			ExpressionBinder::PushCollation(binder.context, cond.RightReference(), cond.GetRHS().return_type);

			join->conditions.push_back(std::move(cond));
		} else { // xm：不是只有一个孩子，说明不是可以拆开单独比较，比如 (a,b) = ANY(x,y)，被分成了 a = x and b = y
			// Standard case: compare each child separately
			for (idx_t child_idx = 0; child_idx < expr.children.size(); child_idx++) {
				auto &child_type = expr.child_types[child_idx];
				auto &compare_type = expr.child_targets[child_idx];
				auto right_expr = BoundCastExpression::AddDefaultCastToType(
				    make_uniq<BoundColumnRefExpression>(child_type, plan_columns[child_idx]), compare_type);
				JoinCondition cond(std::move(expr.children[child_idx]), std::move(right_expr), expr.comparison_type);

				// push collations
				ExpressionBinder::PushCollation(binder.context, cond.LeftReference(), compare_type);
				ExpressionBinder::PushCollation(binder.context, cond.RightReference(), compare_type);

				join->conditions.push_back(std::move(cond));
			}
		}
		root = std::move(join);

		// we replace the original subquery with a BoundColumnRefExpression referring to the mark column
		// xm: mark_index表示的其实是 mark join生成的那一个列
		return make_uniq<BoundColumnRefExpression>(expr.GetName(), expr.return_type,
		                                           ColumnBinding(mark_index, ProjectionIndex(0)));
	}
	}
}

static unique_ptr<LogicalDependentJoin> CreateDuplicateEliminatedJoin(const CorrelatedColumns &correlated_columns,
                                                                      JoinType join_type,
                                                                      unique_ptr<LogicalOperator> original_plan,
                                                                      bool perform_delim) {
	auto delim_join = make_uniq<LogicalDependentJoin>(join_type);
	delim_join->correlated_columns = correlated_columns;
	delim_join->perform_delim = perform_delim;
	delim_join->join_type = join_type;
	delim_join->AddChild(std::move(original_plan));
	// 遍历所有的关联列，构建去重所需的数据结构
	for (idx_t i = 0; i < correlated_columns.size(); i++) {
		auto &col = correlated_columns[i];
		// 把关联列转换为具体的表达式（BoundColumnRefExpression），存入 duplicate_eliminated_columns
		// 物理执行引擎未来会根据这个列表，去左表中提取真实的数据进行去重
		delim_join->duplicate_eliminated_columns.push_back(make_uniq<BoundColumnRefExpression>(col.type, col.binding));
		// 记录这些列的数据类型
		delim_join->mark_types.push_back(col.type);
	}
	return delim_join;
}

static bool PerformDelimOnType(const LogicalType &type) {
	if (type.InternalType() == PhysicalType::LIST) {
		return false;
	}
	if (type.InternalType() == PhysicalType::STRUCT) {
		for (auto &entry : StructType::GetChildTypes(type)) {
			if (!PerformDelimOnType(entry.second)) {
				return false;
			}
		}
	}
	return true;
}

static bool PerformDuplicateElimination(Binder &binder, CorrelatedColumns &correlated_columns) {
	if (!ClientConfig::GetConfig(binder.context).enable_optimizer) {
		// if optimizations are disabled we always do a delim join
		return true;
	}
	bool perform_delim = true;
	for (auto &col : correlated_columns) {
		if (!PerformDelimOnType(col.type)) {
			perform_delim = false;
			break;
		}
	}
	if (perform_delim) {
		return true;
	}
	auto binding = ColumnBinding(binder.GenerateTableIndex(), ProjectionIndex(0));
	auto type = LogicalType::BIGINT;
	auto name = "delim_index";
	CorrelatedColumnInfo info(binding, type, name, 0);
	correlated_columns.AddColumn(std::move(info));
	correlated_columns.SetDelimIndexToZero();
	return false;
}

// xm: 不管是mark join还是single join，join结果总是比左边多出一个列，这个列正是上面filter需要引用的列；
// 对于mark join，这个列是一个布尔列，表示是否匹配；对于single join，是一个值列；
// 多出来的这一列并没有在这里处理，我觉得应该是在后面的列裁剪中统一处理。
static unique_ptr<Expression> PlanCorrelatedSubquery(Binder &binder, BoundSubqueryExpression &expr,
                                                     unique_ptr<LogicalOperator> &root,
                                                     unique_ptr<LogicalOperator> plan) {
	auto &correlated_columns = expr.binder->correlated_columns;
	// FIXME: 应该有一种方法可以禁用 ANY 查询的去相关化，但现在还不能...
	bool perform_delim =
	    expr.subquery_type == SubqueryType::ANY ? true : PerformDuplicateElimination(binder, correlated_columns);
	D_ASSERT(expr.IsCorrelated());
	// 相关子查询 
	// 关于此代码的更深入解释，请阅读论文 "Unnesting Arbitrary Subqueries" 
	// 也可以阅读 "Improving Unnesting of Complex Queries" 
	// 我们处理三种类型的相关子查询：标量（Scalar）、EXISTS 和 ANY 
	// 这三种情况非常相似，只有一些细微的差别（主要是最后执行的连接类型）
	switch (expr.subquery_type) {
	case SubqueryType::SCALAR: { // xm：本来引用的是子查询，现在我们想引用一个标量列
		// 相关标量查询 
		// 首先推送一个去重连接（DUPLICATE ELIMINATED join） 
		// 去重连接会创建 LHS 的去重副本 
		// 并将其推送到 RHS 上的任何去重扫描（DUPLICATE_ELIMINATED SCAN）操作符中

		// 在标量情况下，我们创建一个 SINGLE 连接（因为我们只关心获取值）
		// 在此连接中空值是相等的，因为我们只在相关列上进行连接（NULL values are equal in this join）
		// 例如在查询：SELECT (SELECT 42 FROM integers WHERE i1.i IS NULL LIMIT 1) FROM integers i1;
		// 输入值 NULL 将生成值 42，我们需要将 LHS 上的 NULL 与 RHS 上的 NULL 进行连接
		// 左侧是原始计划
		// 这一侧将被去重并推送到 RHS
		auto delim_join =  // xm: single join和left outer join的区别在于：single join只允许LHS的每一行匹配RHS的最多一行，否则报错
		    CreateDuplicateEliminatedJoin(correlated_columns, JoinType::SINGLE, std::move(root), perform_delim);

		// 我们必须存储执行后续去嵌套所需的所有信息
		delim_join->subquery_type = SubqueryType::SCALAR;
		delim_join->any_join = false;

		auto plan_column = plan->GetColumnBindings().back();
		delim_join->AddChild(std::move(plan));
		root = std::move(delim_join);
		// 最后推送引用连接返回的数据元素的 BoundColumnRefExpression

		// xm: 在这个阶段结束时，逻辑计划树长这样：
		// LogicalDelimJoin (记录了需要去重的列，比如 t1.id)
		//   ├── LHS (外部查询)
		//   └── RHS (内部子查询，此时内部仍然包含未解析的关联引用，比如 t1.id = t2.id)
		// 在这个时刻，DelimGet 还没有诞生。
		return make_uniq<BoundColumnRefExpression>(expr.GetName(), expr.return_type, plan_column); // xm: Join 不会改变底层传上来的 Binding ID
	}
	case SubqueryType::EXISTS: { // xm：本来引用的是子查询，现在我们想引用一个布尔列
		// 相关 EXISTS 查询
		// 此查询与相关标量查询类似，只是这里我们使用 MARK 连接
		auto mark_index = binder.GenerateTableIndex();
		auto delim_join =                                 // xm: mark join会生成一个布尔列
		    CreateDuplicateEliminatedJoin(correlated_columns, JoinType::MARK, std::move(root), perform_delim);

		delim_join->subquery_type = SubqueryType::EXISTS;
		delim_join->mark_index = mark_index;
		delim_join->any_join = true;
		delim_join->AddChild(std::move(plan));
		root = std::move(delim_join);
		// 最后推送引用标记的 BoundColumnRefExpression
		return make_uniq<BoundColumnRefExpression>(expr.GetName(), expr.return_type,
		                                           ColumnBinding(mark_index, ProjectionIndex(0)));
	}
	default: { // 本来引用的是一个子查询，现在我们想引用一个布尔列
		D_ASSERT(expr.subquery_type == SubqueryType::ANY);
		// 相关 ANY 查询
		// 此查询与相关标量查询类似
		// 但是，在这种情况下我们推送一个相关 MARK 连接
		// 注意在此连接中，空值对于所有列都不相等，只对相关列相等
		// 相关标记连接（the correlated mark join）本身会处理这种情况
		// 因为 MARK 连接有一个额外的连接条件（ANY 表达式的原始条件，例如
		// [i=ANY(...)])
		auto mark_index = binder.GenerateTableIndex();
		auto delim_join =
		    CreateDuplicateEliminatedJoin(correlated_columns, JoinType::MARK, std::move(root), perform_delim);

		delim_join->subquery_type = SubqueryType::ANY;
		delim_join->mark_index = mark_index;
		delim_join->any_join = true;
		auto &dependent_join = plan;

		if (expr.children.size() > 1) {
			// FIXME: 这里生成计划的代码实际上是正确的
			// 问题在于哈希连接 - 具体来说是 PhysicalHashJoin::InitializeHashTable
			// 这里包含硬编码用于单个比较的代码
			// -> (delim_types.size() + 1 == conditions.size())
			// 需要将其泛化才能使其工作
			throw NotImplementedException("Correlated IN/ANY/ALL with multiple columns not yet supported");
		}

		delim_join->expression_children = std::move(expr.children);
		delim_join->child_types = expr.child_types;
		delim_join->child_targets = expr.child_targets;
		delim_join->comparison_type = expr.comparison_type;

		delim_join->AddChild(std::move(dependent_join));
		root = std::move(delim_join);
		// 最后推送引用标记的 BoundColumnRefExpression
		return make_uniq<BoundColumnRefExpression>(expr.GetName(), expr.return_type,
		                                           ColumnBinding(mark_index, ProjectionIndex(0)));
	}
	}
}

void RecursiveDependentJoinPlanner::VisitOperator(LogicalOperator &op) {
	if (!op.children.empty()) {
		// Collect all recursive CTEs during recursive descend
		if (op.type == LogicalOperatorType::LOGICAL_RECURSIVE_CTE ||
		    op.type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
			auto &rec_cte = op.Cast<LogicalCTE>();
			binder.recursive_ctes[rec_cte.table_index] = &op;
		}
		for (idx_t i = 0; i < op.children.size(); i++) {
			root = std::move(op.children[i]);
			D_ASSERT(root);
			VisitOperatorExpressions(op);
			op.children[i] = std::move(root);
		}

		for (idx_t i = 0; i < op.children.size(); i++) {
			D_ASSERT(op.children[i]);
			VisitOperator(*op.children[i]);
		}
	}
}

unique_ptr<Expression> RecursiveDependentJoinPlanner::VisitReplace(BoundSubqueryExpression &expr,
                                                                   unique_ptr<Expression> *expr_ptr) {
	return binder.PlanSubquery(expr, root);
}

unique_ptr<Expression> Binder::PlanSubquery(BoundSubqueryExpression &expr, unique_ptr<LogicalOperator> &root) {
	D_ASSERT(root);
	// first we translate the QueryNode of the subquery into a logical plan
	auto sub_binder = Binder::CreateBinder(context, this);
	sub_binder->is_outside_flattened = false;
	auto subquery_root = std::move(expr.subquery.plan); // xm: 已经初步解析好的子查询计划
	D_ASSERT(subquery_root);

	// now we actually flatten the subquery
	auto plan = std::move(subquery_root);

	unique_ptr<Expression> result_expression;
	if (!expr.IsCorrelated()) { // xm: 如果子查询不相关, 如果是标量子查询，就把子查询替换成一个数值列引用，如果是exist或any子查询，就把子查询替换成一个布尔列；
		// xm：看一下Expression中的子查询表示就知道了为什么可以替换以及怎么替换：https://my.feishu.cn/wiki/HZU0wRlgGieS3XkQjojcfp65n8x
		result_expression = PlanUncorrelatedSubquery(*this, expr, root, std::move(plan)); 
	} else { // xm: 如果相关子查询
		result_expression = PlanCorrelatedSubquery(*this, expr, root, std::move(plan));
	}
	IncreaseDepth();
	// finally, we recursively plan the nested subqueries (if there are any)
	if (sub_binder->has_unplanned_dependent_joins) {
		RecursiveDependentJoinPlanner plan(*this);
		plan.VisitOperator(*root);
	}
	return result_expression;
}

// xm: 下面函数做了两件事：
// 1. expr_ptr被替换了
// 2. root被更新了
void Binder::PlanSubqueries(unique_ptr<Expression> &expr_ptr, unique_ptr<LogicalOperator> &root) {
	if (!expr_ptr) { // xm: 没有表达式，直接返回
		return;
	}
	auto &expr = *expr_ptr; // xm: 将智能指针解引用为 expr
	// first visit the children of the node, if any xm: 递归调用自己，遍历当前表达式的所有子表达式。
	ExpressionIterator::EnumerateChildren(expr, [&](unique_ptr<Expression> &expr) { PlanSubqueries(expr, root); });

	// check if this is a subquery node
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_SUBQUERY) {
		auto &subquery = expr.Cast<BoundSubqueryExpression>();
		// subquery node! plan it
		expr_ptr = PlanSubquery(subquery, root);
	}
}

unique_ptr<LogicalOperator> Binder::PlanLateralJoin(unique_ptr<LogicalOperator> left, unique_ptr<LogicalOperator> right,
                                                    CorrelatedColumns &correlated, JoinType join_type,
                                                    unique_ptr<Expression> condition) {
	// scan the right operator for correlated columns
	// correlated LATERAL JOIN
	vector<JoinCondition> conditions;
	if (condition) {
		if (condition->HasSubquery()) {
			throw BinderException(*condition, "Subqueries are not supported in LATERAL join conditions");
		}
		// extract join conditions, if there are any
		LogicalComparisonJoin::ExtractJoinConditions(context, join_type, JoinRefType::REGULAR, left, right,
		                                             std::move(condition), conditions);
	}

	vector<JoinCondition> comparison_conditions;
	vector<unique_ptr<Expression>> non_comparison_conditions;
	for (auto &cond : conditions) {
		if (cond.IsComparison()) {
			comparison_conditions.push_back(std::move(cond));
		} else {
			non_comparison_conditions.push_back(JoinCondition::CreateExpression(std::move(cond)));
		}
	}

	auto perform_delim = PerformDuplicateElimination(*this, correlated);
	auto delim_join = CreateDuplicateEliminatedJoin(correlated, join_type, std::move(left), perform_delim);

	// Store all information required to perform UNNESTING later.
	delim_join->perform_delim = perform_delim;
	delim_join->any_join = false;
	delim_join->propagate_null_values = join_type != JoinType::INNER;
	delim_join->is_lateral_join = true;
	delim_join->arbitrary_expressions = std::move(non_comparison_conditions);
	delim_join->conditions = std::move(comparison_conditions);
	delim_join->AddChild(std::move(right));
	return std::move(delim_join);
}

} // namespace duckdb
