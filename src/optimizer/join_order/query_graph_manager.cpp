#include "duckdb/optimizer/join_order/query_graph_manager.hpp"

#include "duckdb/common/assert.hpp"
#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/optimizer/join_order/join_relation.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/list.hpp"

namespace duckdb {

//! Returns true if A and B are disjoint, false otherwise
template <class T>
static bool Disjoint(const unordered_set<T> &a, const unordered_set<T> &b) {
	return std::all_of(a.begin(), a.end(), [&b](typename std::unordered_set<T>::const_reference entry) {
		return b.find(entry) == b.end();
	});
}

bool QueryGraphManager::Build(JoinOrderOptimizer &optimizer, LogicalOperator &op) {
	// 让关系管理器提取连接关系，并创建所有过滤操作符的引用列表。
	auto can_reorder = relation_manager.ExtractJoinRelations(optimizer, op, filter_operators);
	auto num_relations = relation_manager.NumRelations();
	if (num_relations <= 1 || !can_reorder) {
		// nothing to optimize/reorder
		return false;
	}
	// 提取超图的边，创建过滤器列表及其相关绑定。
	filters_and_bindings = relation_manager.ExtractEdges(op, filter_operators, set_manager);
	// Create the query_graph hyper edges
	CreateHyperGraphEdges();
	return true;
}

void QueryGraphManager::GetColumnBinding(Expression &root_expr, ColumnBinding &binding) {
	ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
	    root_expr, [&](const BoundColumnRefExpression &colref) {
		    D_ASSERT(colref.depth == 0);
		    D_ASSERT(colref.binding.table_index.IsValid());
		    // map the base table index to the relation index used by the JoinOrderOptimizer
		    D_ASSERT(relation_manager.relation_mapping.find(colref.binding.table_index) !=
		             relation_manager.relation_mapping.end());
		    binding = ColumnBinding(TableIndex(relation_manager.relation_mapping[colref.binding.table_index].index),
		                            colref.binding.column_index);
	    });
}

const vector<unique_ptr<FilterInfo>> &QueryGraphManager::GetFilterBindings() const {
	return filters_and_bindings;
}

void FilterInfo::SetLeftSet(optional_ptr<JoinRelationSet> left_set_new) {
	left_set = left_set_new;
}

void FilterInfo::SetRightSet(optional_ptr<JoinRelationSet> right_set_new) {
	right_set = right_set_new;
}

static unique_ptr<LogicalOperator> PushFilter(unique_ptr<LogicalOperator> node, unique_ptr<Expression> expr) {
	// push an expression into a filter
	// first check if we have any filter to push it into
	if (node->type != LogicalOperatorType::LOGICAL_FILTER) {
		// we don't, we need to create one
		auto filter = make_uniq<LogicalFilter>();
		filter->children.push_back(std::move(node));
		node = std::move(filter);
	}
	// push the filter into the LogicalFilter
	D_ASSERT(node->type == LogicalOperatorType::LOGICAL_FILTER);
	auto &filter = node->Cast<LogicalFilter>();
	filter.expressions.push_back(std::move(expr));
	return node;
}

// 建边的时候是有条件的，并不是filters_and_bindings中的所有条件都可以建成边
void QueryGraphManager::CreateHyperGraphEdges() {
	// create potential edges from the comparisons
	for (auto &filter_info : filters_and_bindings) { // 前面刚收集的 filter_info
		auto &filter = filter_info->filter;
		// now check if it can be used as a join predicate
		if (filter->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON) { // 当前表达式是一个比较表达式，不是 “and”
			auto &comparison = filter->Cast<BoundComparisonExpression>();
			// 提取比较操作左右两侧所需的绑定（bindings）。
			unordered_set<RelationIndex> left_bindings, right_bindings;
			relation_manager.ExtractBindings(*comparison.left, left_bindings);
			relation_manager.ExtractBindings(*comparison.right, right_bindings);
			GetColumnBinding(*comparison.left, filter_info->left_binding);
			GetColumnBinding(*comparison.right, filter_info->right_binding);
			if (!left_bindings.empty() && !right_bindings.empty()) {
				// 左侧和右侧都存在绑定（bindings）
				// 首先创建关系集合（relation sets），如果它们尚不存在的话
				if (!filter_info->left_set) {
					filter_info->left_set = &set_manager.GetJoinRelation(left_bindings);
				}
				if (!filter_info->right_set) {
					filter_info->right_set = &set_manager.GetJoinRelation(right_bindings);
				}
				// 只有当这两个集合不完全相同时，我们才能创建一条有意义的边
				if (filter_info->left_set != filter_info->right_set) {
					// 进一步检查这两个集合是否不相交（即没有公共元素）
					if (Disjoint(left_bindings, right_bindings)) {
						// 它们是不相交的，我们只需在连接图中创建一组边
						query_graph.CreateEdge(*filter_info->left_set, *filter_info->right_set, filter_info); // 两个单向边构成一个双向边
						query_graph.CreateEdge(*filter_info->right_set, *filter_info->left_set, filter_info);
					}
				}
			}
		} else if (filter->GetExpressionClass() == ExpressionClass::BOUND_CONJUNCTION) {
			auto &conjunction = filter->Cast<BoundConjunctionExpression>();
			if (conjunction.GetExpressionType() == ExpressionType::CONJUNCTION_OR ||
			    filter_info->join_type == JoinType::INNER || filter_info->join_type == JoinType::INVALID) {
				// 目前我们不会将合取（Conjunction）表达式解释为超图边的内连接（INNER joins）。
				// 这些合取表达式很可能是“OR”类型的合取，并将在优化器后续阶段下推到连接操作中。
				// 目前，合取过滤条件主要用于辅助规划半连接（semi joins）和反连接（anti joins）。
				continue;
			}
			unordered_set<RelationIndex> left_bindings, right_bindings;
			D_ASSERT(filter_info->left_set);
			D_ASSERT(filter_info->right_set); // SEMI/ANTI 的 AND 享受 VIP 待遇
			D_ASSERT(filter_info->join_type == JoinType::SEMI || filter_info->join_type == JoinType::ANTI); // 要么是半连接，要么是反连接
			for (auto &child_comp : conjunction.children) {
				if (child_comp->GetExpressionClass() != ExpressionClass::BOUND_COMPARISON) {
					continue;
				}
				auto &comparison = child_comp->Cast<BoundComparisonExpression>();
				// 提取比较操作左侧和右侧所需的绑定（bindings）
				relation_manager.ExtractBindings(*comparison.left, left_bindings);
				relation_manager.ExtractBindings(*comparison.right, right_bindings);
				if (!filter_info->left_binding.table_index.IsValid() &&
				    !filter_info->left_binding.column_index.IsValid()) {
					GetColumnBinding(*comparison.left, filter_info->left_binding);
				}
				if (!filter_info->right_binding.table_index.IsValid() &&
				    !filter_info->right_binding.column_index.IsValid()) {
					GetColumnBinding(*comparison.right, filter_info->right_binding);
				}
			}
			if (!left_bindings.empty() && !right_bindings.empty()) {
				// 只有当两个集合不完全相同时，我们才能创建一条有意义的边
				if (filter_info->left_set != filter_info->right_set) {
					// 检查两个集合是否不相交
					if (Disjoint(left_bindings, right_bindings)) {
						// 它们是不相交的，我们只需在连接图中创建一组边
						query_graph.CreateEdge(*filter_info->left_set, *filter_info->right_set, filter_info);
						query_graph.CreateEdge(*filter_info->right_set, *filter_info->left_set, filter_info);
					}
				}
			}
		}
	}
}

// 根据rel中的op，从old plan中“挖”出来一块逻辑计划，并返回它的所有权
static unique_ptr<LogicalOperator> ExtractJoinRelation(unique_ptr<SingleJoinRelation> &rel) {
	auto &children = rel->parent->children; // 拿到rel的父节点的子节点“列表”
	for (idx_t i = 0; i < children.size(); i++) {
		if (children[i].get() == &rel->op) {
			// found it! take ownership o/**/f it from the parent
			auto result = std::move(children[i]); // 查询树的节点是用 std::unique_ptr（独占智能指针）管理的。
			children.erase_at(i);
			return result;
		}
	}
	throw InternalException("Could not find relation in parent node (?)");
}

unique_ptr<LogicalOperator> QueryGraphManager::Reconstruct(unique_ptr<LogicalOperator> plan) {
	// now we have to rewrite the plan
	bool root_is_join = plan->children.size() > 1;

	unordered_set<RelationIndex> bindings;
	for (idx_t i = 0; i < relation_manager.NumRelations(); i++) {
		bindings.emplace(i);
	}
	auto &total_relation = set_manager.GetJoinRelation(bindings);
	// xm: 从这个地方可以推断出来，relation_manager.NumRelations()代表本次动态规划中“表”的数量

	// first we will extract all relations from the main plan
	vector<unique_ptr<LogicalOperator>> extracted_relations;
	extracted_relations.reserve(relation_manager.NumRelations()); // 为 vector 预分配空间，大小为关系数量
	for (auto &relation : relation_manager.GetRelations()) {  // 遍历DPHyp的所有“基”表，基SingleJoinRelation
		extracted_relations.push_back(ExtractJoinRelation(relation));
	}

	// now we generate the actual joins
	auto join_tree = GenerateJoins(extracted_relations, total_relation);

	// perform the final pushdown of remaining filters
	for (auto &filter : filters_and_bindings) {
		// check if the filter has already been extracted
		if (filter->filter) {
			// if not we need to push it
			join_tree.op = PushFilter(std::move(join_tree.op), std::move(filter->filter));
		}
	}

	// find the first join in the relation to know where to place this node
	if (root_is_join) {
		// first node is the join, return it immediately
		return std::move(join_tree.op);
	}
	D_ASSERT(plan->children.size() == 1);
	// have to move up through the relations
	auto op = plan.get();
	auto parent = plan.get();
	while (op->type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT &&
	       op->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
	       op->type != LogicalOperatorType::LOGICAL_ASOF_JOIN) {
		D_ASSERT(op->children.size() == 1);
		parent = op;
		op = op->children[0].get();
	}
	// have to replace at this node
	parent->children[0] = std::move(join_tree.op);
	return plan;
}

static JoinCondition MaybeInvertConditions(unique_ptr<Expression> condition, bool invert) {
	auto &comparison = condition->Cast<BoundComparisonExpression>();
	auto left = !invert ? std::move(comparison.left) : std::move(comparison.right);
	auto right = !invert ? std::move(comparison.right) : std::move(comparison.left);
	auto comp_type = condition->GetExpressionType();
	if (invert) {
		// reverse comparison expression if we reverse the order of the children
		comp_type = FlipComparisonExpression(comp_type);
	}
	return JoinCondition(std::move(left), std::move(right), comp_type);
}

// xm: 拿着动态规划（DP）算出来的最优结构图，通过递归的方式，把底层的单表（砖块）一层一层地拼装成一棵
// 可以实际执行的物理/逻辑算子树（AST），同时在这个过程中把所有的过滤条件（WHERE/ON）精准地安插到正确的位置。

// 动态规划图纸上的谓词有什么特点？
// 首先，它是一条边，也就是说，既涉及左边，又涉及右边；这就保证了它一定不会被下推；
// 其次，图纸上的边组合在一起，未必就是全部的谓词，因为我的建的边是有冗余的；这就需要我们去主动找一下该Join节点上能够使用的谓词；
// debug: -exec call filters_and_bindings[2]->filter->ToString()
GenerateJoinRelation QueryGraphManager::GenerateJoins(vector<unique_ptr<LogicalOperator>> &extracted_relations,
                                                      JoinRelationSet &set) {
	optional_ptr<JoinRelationSet> left_node;
	optional_ptr<JoinRelationSet> right_node;
	optional_ptr<JoinRelationSet> result_relation;
	unique_ptr<LogicalOperator> result_operator;

	auto dp_entry = plans->find(set);
	if (dp_entry == plans->end()) {
		throw InternalException("Join Order Optimizer Error: No full plan was created");
	}
	auto &node = dp_entry->second; // duckdb::DPJoinNode
	if (!dp_entry->second->is_leaf) {
		// generate the left and right children
		auto left = GenerateJoins(extracted_relations, node->left_set); // 处理左子树
		auto right = GenerateJoins(extracted_relations, node->right_set); // 处理右子树
		if (dp_entry->second->info->filters.empty()) {
			// no filters, create a cross product
			auto cardinality = left.op->estimated_cardinality * right.op->estimated_cardinality;
			result_operator = LogicalCrossProduct::Create(std::move(left.op), std::move(right.op));
			result_operator->SetEstimatedCardinality(cardinality);
		} else {
			// we have filters, create a join node, xm：同时确定连接类型
			auto chosen_filter = node->info->filters.at(0); // 遍历一条边的多个谓词
			for (idx_t i = 0; i < node->info->filters.size(); i++) {
				if (node->info->filters.at(i)->join_type == JoinType::INNER) { // xm:INNER JOIN 优先级最高
					chosen_filter = node->info->filters.at(i);
					break;
				}
			}
			auto join = make_uniq<LogicalComparisonJoin>(chosen_filter->join_type);
			// 这里我们优化构建侧和探测侧。我们的构建侧是右侧
			// 因此右侧计划应该具有更低的基数。
			join->children.push_back(std::move(left.op));
			join->children.push_back(std::move(right.op));

			// 从连接节点设置连接条件
			for (auto &filter_ref : node->info->filters) {
				auto f = filter_ref.get();
				// 从过滤器原本所属的算子中提取该过滤器
				D_ASSERT(filters_and_bindings[f->filter_index]->filter); // 一定不为空，为什么？
				auto &filter_and_binding = filters_and_bindings.at(f->filter_index);
				auto condition = std::move(filter_and_binding->filter);
				// 现在创建实际的连接条件
				D_ASSERT((JoinRelationSet::IsSubset(*left.set, *f->left_set) &&
				          JoinRelationSet::IsSubset(*right.set, *f->right_set)) ||
				         (JoinRelationSet::IsSubset(*left.set, *f->right_set) &&
				          JoinRelationSet::IsSubset(*right.set, *f->left_set)));

				bool invert = !JoinRelationSet::IsSubset(*left.set, *f->left_set);
				// 如果左右集合被反转了并且这是一个半连接或反连接
				// 则将左右子节点交换回来。（以谓词的左右为基准，因为我们之前没有反转过谓词）

				if (invert && (f->join_type == JoinType::SEMI || f->join_type == JoinType::ANTI)) {
					std::swap(join->children[0], join->children[1]);
					invert = false;
				} // inner join不需要交换，但是可能需要反转谓词，以谓词谓词和连接方向的对应

				if (condition->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON) {  // 简单表达式
					auto cond = MaybeInvertConditions(std::move(condition), invert); // 保持表达式和连接方向的一致性
					join->conditions.push_back(std::move(cond));
				} else if (condition->GetExpressionClass() == ExpressionClass::BOUND_CONJUNCTION) {
					auto &conjunction = condition->Cast<BoundConjunctionExpression>();
					for (auto &child : conjunction.children) {
						D_ASSERT(child->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON); // AND连接的复杂表达式
						auto cond = MaybeInvertConditions(std::move(child), invert);
						join->conditions.push_back(std::move(cond));
					}
				}
			}
			D_ASSERT(!join->conditions.empty());
			result_operator = std::move(join);
		}
		left_node = left.set;
		right_node = right.set;
		result_relation = &set_manager.Union(*left.set, *right.set);
	} else {
		// base node, get the entry from the list of extracted relations
		D_ASSERT(node->set.count == 1);
		D_ASSERT(extracted_relations[node->set.relations[0].index]);
		result_relation = &node->set;
		result_operator = std::move(extracted_relations[result_relation->relations[0].index]);
	}
	// TODO（待办事项）：这里是“估算属性（estimated properties）”开始发挥作用的地方。
	// 当创建结果算子（result operator）时，我们应该向代价模型（cost model）和基数估算器（cardinality estimator）查询
	// 它的代价（cost）和基数（cardinality）分别是多少
	// result_operator->estimated_props = node.estimated_props->Copy();
	result_operator->estimated_cardinality = node->cardinality;
	result_operator->has_estimated_cardinality = true;

	// 收集属于当前这个 Join 的、尚未被使用的残余谓词
	vector<unique_ptr<Expression>> unused_residual_predicates;
	for (auto &filter_info : filters_and_bindings) {
		if (filter_info->from_residual_predicate && filters_and_bindings[filter_info->filter_index]->filter) {
			// xm: 遍历所有目前还没有被使用的残余谓词（residual predicates），能用则用
			if (filter_info->set.get().count > 0 && JoinRelationSet::IsSubset(*result_relation, filter_info->set)) {
				unused_residual_predicates.push_back(
				    std::move(filters_and_bindings[filter_info->filter_index]->filter));
			}
		}
	}

	if (!unused_residual_predicates.empty()) {
		// 把收集到的残余谓词组合成一个大的 AND 表达式，称为 combined
		unique_ptr<Expression> combined = std::move(unused_residual_predicates[0]);
		for (idx_t i = 1; i < unused_residual_predicates.size(); i++) {
			combined = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND, std::move(combined),
			                                                 std::move(unused_residual_predicates[i]));
		}

		if (result_operator->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
			// attach to join's predicate field // xm: 把 combined 这个残余谓词附加到连接的谓词字段上
			auto &comp_join = result_operator->Cast<LogicalComparisonJoin>();
			comp_join.conditions.emplace_back(std::move(combined));
		} else {
			// push as filter // xm: 可能是笛卡尔积，需要生成一个过滤器把 combined 这个残余谓词挂上去
			result_operator = PushFilter(std::move(result_operator), std::move(combined));
		}
	}

	// 检查我们是否应该在当前节点执行（过滤条件）下推
	// 基本上，如果任何剩余的过滤条件（remaining filter）（所需的表）已经是当前节点集合的子集，那么这些条件在后续的 Join 中就再也派不上用场了
	// 因此，我们应该（趁早）在这里把它推入（应用掉）
	for (auto &filter_info : filters_and_bindings) {
		// check if the filter has already been extracted
		auto &info = *filter_info;
		if (filters_and_bindings[info.filter_index]->filter) {
			// skip filters from residual predicates // 下面：因为前面已经处理过了
			if (info.from_residual_predicate) {
				continue;
			}

			// 现在检查该过滤条件（所需的表集合）是否是当前关系集合的子集
			// 请注意，关系集合为空（empty relation set）的 info（过滤条件信息）是一个特殊情况，我们不会将它们下推
			if (info.set.get().count > 0 && JoinRelationSet::IsSubset(*result_relation, info.set)) {
				auto &filter_and_binding = filters_and_bindings[info.filter_index];
				auto filter = std::move(filter_and_binding->filter);
				// 如果是的话（即该条件所需的表已经是当前集合的子集），我们就可以下推这个过滤条件
				// 我们可以把它直接推入（融合）到一个 Join 算子内部，或者作为一个独立的 Filter 算子来下推
				// 检查我们当前究竟是在处理一个 Join 节点，还是在一个基础表（Base Table）上
				if (!left_node || !info.left_set) {
					// base table or non-comparison expression, push it as a filter
					result_operator = PushFilter(std::move(result_operator), std::move(filter));
					continue;
				}
				// 我们正下方（当前处理）的节点是一个 Join 或是笛卡尔积（Cross Product），并且该表达式是一个比较操作
				// 检查（该比较条件涉及的表集合）是否可以被清晰地拆分，并分别对应到当前算子的左、右子树上
				bool found_subset = false;
				bool invert = false;
				if (JoinRelationSet::IsSubset(*left_node, *info.left_set) &&
				    JoinRelationSet::IsSubset(*right_node, *info.right_set)) {
					found_subset = true;
				} else if (JoinRelationSet::IsSubset(*right_node, *info.left_set) &&
				           JoinRelationSet::IsSubset(*left_node, *info.right_set)) {
					invert = true;
					found_subset = true;
				}
				if (!found_subset) {
					// could not be split up into left/right
					result_operator = PushFilter(std::move(result_operator), std::move(filter));
					continue;
				}
				// create the join condition
				D_ASSERT(filter->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON);
				auto &comparison = filter->Cast<BoundComparisonExpression>();
				// 我们需要通过查看当前可用的关系（表集合），来分辨出（表达式的）哪一边对应（算子的）哪一边
				auto left = !invert ? std::move(comparison.left) : std::move(comparison.right);
				auto right = !invert ? std::move(comparison.right) : std::move(comparison.left);
				auto comp_type = comparison.GetExpressionType();
				if (invert) {
					// reverse comparison expression if we reverse the order of the children
					comp_type = FlipComparisonExpression(comp_type);
				}
				JoinCondition cond(std::move(left), std::move(right), comp_type);
				// now find the join to push it into xm: 此时result_operator要么是join，要么是作为join父节点的一个filter
				auto node = result_operator.get();
				if (node->type == LogicalOperatorType::LOGICAL_FILTER) {
					node = node->children[0].get();
				}
				// xm: 现在已经保证node是一个连接
				if (node->type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
					// turn into comparison join // xm: 笛卡尔积变内连接
					auto comp_join = make_uniq<LogicalComparisonJoin>(JoinType::INNER);
					comp_join->children.push_back(std::move(node->children[0]));
					comp_join->children.push_back(std::move(node->children[1]));
					comp_join->conditions.push_back(std::move(cond));
					if (node == result_operator.get()) { // node 就是 result_operator，即result_operator是一个连接
						result_operator = std::move(comp_join); // 替换原来的笛卡尔积
					} else {
						D_ASSERT(result_operator->type == LogicalOperatorType::LOGICAL_FILTER);
						result_operator->children[0] = std::move(comp_join); // 替换原来的笛卡尔积
					}
				} else { // 本就是“连接”，把这个条件直接附加到连接的谓词字段上
					D_ASSERT(node->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
					         node->type == LogicalOperatorType::LOGICAL_ASOF_JOIN);
					auto &comp_join = node->Cast<LogicalComparisonJoin>();
					comp_join.conditions.push_back(std::move(cond));
				}
			}
		}
	}
	auto result = GenerateJoinRelation(result_relation, std::move(result_operator));
	return result;
}

const QueryGraphEdges &QueryGraphManager::GetQueryGraphEdges() const {
	return query_graph;
}

void QueryGraphManager::CreateQueryGraphCrossProduct(JoinRelationSet &left, JoinRelationSet &right) {
	query_graph.CreateEdge(left, right, nullptr);
	query_graph.CreateEdge(right, left, nullptr);
}

} // namespace duckdb
