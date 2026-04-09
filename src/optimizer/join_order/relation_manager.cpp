#include "duckdb/optimizer/join_order/relation_manager.hpp"

#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/common/enums/logical_operator_type.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/optimizer/join_order/join_order_optimizer.hpp"
#include "duckdb/optimizer/join_order/relation_statistics_helper.hpp"
#include "duckdb/parser/expression_map.hpp"
#include "duckdb/planner/expression/list.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/list.hpp"

namespace duckdb {

const vector<RelationStats> RelationManager::GetRelationStats() {
	vector<RelationStats> ret;
	for (idx_t i = 0; i < relations.size(); i++) {
		ret.push_back(relations[i]->stats);
	}
	return ret;
}

vector<unique_ptr<SingleJoinRelation>> RelationManager::GetRelations() {
	return std::move(relations);
}

idx_t RelationManager::NumRelations() {
	return relations.size();
}

void RelationManager::AddAggregateOrWindowRelation(LogicalOperator &op, optional_ptr<LogicalOperator> parent,
                                                   const RelationStats &stats, LogicalOperatorType op_type) {
	auto relation = make_uniq<SingleJoinRelation>(op, parent, stats);
	RelationIndex relation_id(relations.size());

	auto op_bindings = op.GetColumnBindings();
	for (auto &binding : op_bindings) {
		if (relation_mapping.find(binding.table_index) == relation_mapping.end()) {
			relation_mapping[binding.table_index] = relation_id;
		}
	}
	relations.push_back(std::move(relation));
	op.estimated_cardinality = stats.cardinality;
	op.has_estimated_cardinality = true;
}

// xm comment: 1.把op及其子树包装成一个relation;
// 2.把这个relation中引用的所有表都添加到relation mapping中;
void RelationManager::AddRelation(LogicalOperator &op, optional_ptr<LogicalOperator> parent,
                                  const RelationStats &stats) {
	// if parent is null, then this is a root relation
	// if parent is not null, it should have multiple children
	D_ASSERT(!parent || parent->children.size() >= 2);
	auto relation = make_uniq<SingleJoinRelation>(op, parent, stats); // 包装成relation
	RelationIndex relation_id(relations.size()); // 分配超图中的编号

	auto table_indexes = op.GetTableIndex();
	bool is_mark = op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
	               op.Cast<LogicalComparisonJoin>().join_type == JoinType::MARK;
	bool get_all_child_bindings = op.type == LogicalOperatorType::LOGICAL_UNNEST;
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		get_all_child_bindings = !op.children.empty();
	}
	if (table_indexes.empty() || is_mark) {
		// relation 表示一个不可重排序的关系，很可能是一个连接（join）关系。
		// 获取该不可重排序关系中引用的所有表，并将它们添加到关系映射（relation mapping）中。
		// 这里应当包含所有的表引用，即使其中存在嵌套的不可重排序连接。
		unordered_set<TableIndex> table_references;
		LogicalJoin::GetTableReferences(op, table_references);
		D_ASSERT(!table_references.empty());
		for (auto &reference : table_references) {
			D_ASSERT(relation_mapping.find(reference) == relation_mapping.end());
			relation_mapping[reference] = relation_id;
		}
	} else if (get_all_child_bindings) {
		// LogicalGet 拥有一个 logical_get 索引，
		// 但如果存在函数（function），其他绑定（bindings）可能会引用那些未被展开（unnested）的列，以及 LogicalGet 子节点中的列。

		// 1. GetColumnBindings(): 拿到这个算子向外输出的**所有**列信息（包括它继承的，以及它新生成的）
		auto bindings = op.GetColumnBindings();
		for (auto &binding : bindings) { // 2. 遍历这些列
			relation_mapping[binding.table_index] = relation_id; // 3. 把这些列所属的所有的 table_index，统统指向当前这一个图节点！
		}
	} else { // 普通表
		// Relations should never return more than 1 table index
		D_ASSERT(table_indexes.size() == 1);
		auto table_index = table_indexes.at(0);
		D_ASSERT(relation_mapping.find(table_index) == relation_mapping.end());
		relation_mapping[table_index] = relation_id;
	}
	relations.push_back(std::move(relation));
	op.estimated_cardinality = stats.cardinality;
	op.has_estimated_cardinality = true;
}

bool RelationManager::CrossProductWithRelationAllowed(idx_t relation_id) {
	return no_cross_product_relations.find(relation_id) == no_cross_product_relations.end();
}

// xm comment: 这个函数的含义是：需不需要把当前这个op当成一个relaton来处理？这里的relation是指超图中的一个普通节点。
static bool OperatorNeedsRelation(LogicalOperatorType op_type) {
	switch (op_type) {
	case LogicalOperatorType::LOGICAL_PROJECTION: // 改变了列，比如生成了新的列；所以把这个操作符及其子树当成一个relation来处理；
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET: // 数据源，本就是relation
	case LogicalOperatorType::LOGICAL_GET: // 数据源，本就是relation
	case LogicalOperatorType::LOGICAL_UNNEST: // unnest会生成新的列，所以把这个操作符及其子树当成一个relation来处理；
	case LogicalOperatorType::LOGICAL_DELIM_GET: // 数据源，本就是relation
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: // 聚合操作，需要当成一个relation来处理
	case LogicalOperatorType::LOGICAL_WINDOW: // 窗口操作，需要当成一个relation来处理
	case LogicalOperatorType::LOGICAL_SAMPLE: // sample会改变行数，所以把这个操作符及其子树当成一个relation来处理；
		return true;
	default:
		return false;
	}
}

// 在LogicalOperatorType层面指明哪一些操作符是不能被重排序的
static bool OperatorIsNonReorderable(LogicalOperatorType op_type) {
	switch (op_type) {
	case LogicalOperatorType::LOGICAL_UNION: // 并集
	case LogicalOperatorType::LOGICAL_EXCEPT: // 差集
	case LogicalOperatorType::LOGICAL_INTERSECT: // 交集
	case LogicalOperatorType::LOGICAL_ANY_JOIN: // 两个特殊的连接类型，不能被重排序
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		return true;
	default:
		return false;
	}
}

bool ExpressionContainsColumnRef(const Expression &root_expr) {
	bool contains_column_ref = false;
	ExpressionIterator::VisitExpression<BoundColumnRefExpression>(root_expr,
	                                                              [&](const BoundColumnRefExpression &colref) {
	// Here you have a filter on a single column in a table. Return a binding for the column
	// being filtered on so the filter estimator knows what HLL count to pull
#ifdef DEBUG
		                                                              (void)colref.depth;
		                                                              D_ASSERT(colref.depth == 0);
		                                                              D_ASSERT(colref.binding.table_index.IsValid());
#endif
		                                                              // map the base table index to the relation index
		                                                              // used by the JoinOrderOptimizer
		                                                              contains_column_ref = true;
	                                                              });
	return contains_column_ref;
}

static bool JoinIsReorderable(LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
		return true;
	}

	if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		auto &join = op.Cast<LogicalComparisonJoin>();

		// TODO: SEMI/ANTI joins with residual predicates are not supported
		if (join.join_type == JoinType::SEMI || join.join_type == JoinType::ANTI) {
			for (auto &cond : join.conditions) {
				if (!cond.IsComparison()) {
					return false;
				}
			}
		}

		switch (join.join_type) {
		case JoinType::INNER:
		case JoinType::SEMI:
		case JoinType::ANTI:
			for (auto &cond : join.conditions) {
				if (cond.IsComparison() && ExpressionContainsColumnRef(cond.GetLHS()) &&
				    ExpressionContainsColumnRef(cond.GetRHS())) {
					return true;
				}
			}
			return false;
		default:
			return false;
		}
	}
	return false;
}

static bool HasNonReorderableChild(LogicalOperator &op) {
	LogicalOperator *tmp = &op;
	while (tmp->children.size() == 1) {
		if (OperatorNeedsRelation(tmp->type) || OperatorIsNonReorderable(tmp->type)) {
			return true;
		}
		tmp = tmp->children[0].get();
		if (tmp->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
			if (!JoinIsReorderable(*tmp)) {
				return true;
			}
		}
	}
	return tmp->children.empty();
}

static void ModifyStatsIfLimit(optional_ptr<LogicalOperator> limit_op, RelationStats &stats) {
	if (!limit_op) {
		return;
	}
	auto &limit = limit_op->Cast<LogicalLimit>();
	if (limit.limit_val.Type() == LimitNodeType::CONSTANT_VALUE) {
		stats.cardinality = MinValue(limit.limit_val.GetConstantValue(), stats.cardinality);
	}
}

void RelationManager::AddRelationWithChildren(JoinOrderOptimizer &optimizer, LogicalOperator &op,
                                              LogicalOperator &input_op, optional_ptr<LogicalOperator> parent,
                                              RelationStats &child_stats, optional_ptr<LogicalOperator> limit_op,
                                              vector<reference<LogicalOperator>> &datasource_filters) {
	D_ASSERT(!op.children.empty());
	auto child_optimizer = optimizer.CreateChildOptimizer();
	op.children[0] = child_optimizer.Optimize(std::move(op.children[0]), &child_stats);
	if (!datasource_filters.empty()) {
		child_stats.cardinality = LossyNumericCast<idx_t>(static_cast<double>(child_stats.cardinality) *
		                                                  RelationStatisticsHelper::DEFAULT_SELECTIVITY);
	}
	ModifyStatsIfLimit(limit_op.get(), child_stats);
	AddRelation(input_op, parent, child_stats);
}

bool RelationManager::ExtractJoinRelations(JoinOrderOptimizer &optimizer, LogicalOperator &input_op,
                                           vector<reference<LogicalOperator>> &filter_operators,
                                           optional_ptr<LogicalOperator> parent) {
	optional_ptr<LogicalOperator> op = &input_op;
	vector<reference<LogicalOperator>> datasource_filters;
	optional_ptr<LogicalOperator> limit_op = nullptr;
	// pass through single child operators xm comment: 穿过单节点操作符，且这些单节点操作符不需要被当成一个relation
	while (op->children.size() == 1 && !OperatorNeedsRelation(op->type)) {
		if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
			if (HasNonReorderableChild(*op)) { // 判断这个filter算子是否被视为一个relation
				datasource_filters.push_back(*op); // 如果被视为一个relation了，就把它加入datasource_filters列表中，后续在计算relation的stats时会用到这个列表中的filter来调整stats
			}
			filter_operators.push_back(*op);
		}
		if (op->type == LogicalOperatorType::LOGICAL_LIMIT) {
			limit_op = op;
		}
		op = op->children[0].get();
	}
	bool non_reorderable_operation = false;
	if (OperatorIsNonReorderable(op->type)) {
		// set operation, optimize separately in children
		non_reorderable_operation = true;
	}
	//上面的OperatorIsNonReorderable没有考虑Join的情况，所以单独把Join拿出来判断了一下
	if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		if (JoinIsReorderable(*op)) {
			// extract join conditions from inner join
			filter_operators.push_back(*op);
		} else {
			non_reorderable_operation = true;
		}
	}
	if (non_reorderable_operation) {
		// 我们遇到了一个不可重排序的操作（如集合操作或非内连接）。
		// 目前我们还不会对非内连接进行重排序，
		// 但我们希望在其周围扩展可能的连接图（join graph）。
		// 非内连接也较为棘手，因为我们不能随意将条件推过它们。
		// 例如，假设我们有：
		//   (left LEFT OUTER JOIN right WHERE right IS NOT NULL)
		// 这个左外连接可能会在右表一侧生成新的 NULL 值，
		// 因此如果将 "right IS NOT NULL" 这个条件下推穿过该连接，
		// 就会导致错误的结果。
		// 出于这个原因，我们改为对这个连接的每个子节点分别启动一次新的 JoinOptimizer 优化过程。
		// 此时，统计信息中的基数（stats.cardinality）将被初始化为各子节点中最大的基数。
		vector<RelationStats> children_stats;
		for (auto &child : op->children) {
			auto stats = RelationStats();
			auto child_optimizer = optimizer.CreateChildOptimizer();
			child = child_optimizer.Optimize(std::move(child), &stats); // 分别优化每一个子节点
			children_stats.push_back(stats);
		}

		auto combined_stats = RelationStatisticsHelper::CombineStatsOfNonReorderableOperator(*op, children_stats);
		op->SetEstimatedCardinality(combined_stats.cardinality);
		if (!datasource_filters.empty()) {
			combined_stats.cardinality = (idx_t)MaxValue(
			    double(combined_stats.cardinality) * RelationStatisticsHelper::DEFAULT_SELECTIVITY, (double)1);
		}
		AddRelation(input_op, parent, combined_stats);
		return true;
	}

	switch (op->type) {
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		// optimize children
		RelationStats child_stats;
		auto child_optimizer = optimizer.CreateChildOptimizer();
		op->children[0] = child_optimizer.Optimize(std::move(op->children[0]), &child_stats);
		auto &aggr = op->Cast<LogicalAggregate>();
		auto operator_stats = RelationStatisticsHelper::ExtractAggregationStats(aggr, child_stats);
		// the extracted cardinality should be set for aggregate
		aggr.SetEstimatedCardinality(operator_stats.cardinality);
		if (!datasource_filters.empty()) {
			operator_stats.cardinality = LossyNumericCast<idx_t>(static_cast<double>(operator_stats.cardinality) *
			                                                     RelationStatisticsHelper::DEFAULT_SELECTIVITY);
		}
		ModifyStatsIfLimit(limit_op.get(), child_stats);
		AddAggregateOrWindowRelation(input_op, parent, operator_stats, op->type);
		return true;
	}
	case LogicalOperatorType::LOGICAL_WINDOW: {
		// optimize children
		RelationStats child_stats;
		auto child_optimizer = optimizer.CreateChildOptimizer();
		op->children[0] = child_optimizer.Optimize(std::move(op->children[0]), &child_stats);
		auto &window = op->Cast<LogicalWindow>();
		auto operator_stats = RelationStatisticsHelper::ExtractWindowStats(window, child_stats);
		// the extracted cardinality should be set for window
		window.SetEstimatedCardinality(operator_stats.cardinality);
		if (!datasource_filters.empty()) {
			operator_stats.cardinality = LossyNumericCast<idx_t>(static_cast<double>(operator_stats.cardinality) *
			                                                     RelationStatisticsHelper::DEFAULT_SELECTIVITY);
		}
		ModifyStatsIfLimit(limit_op.get(), child_stats);
		AddAggregateOrWindowRelation(input_op, parent, operator_stats, op->type);
		return true;
	}
	case LogicalOperatorType::LOGICAL_UNNEST: {
		// optimize children of unnest
		RelationStats child_stats;
		AddRelationWithChildren(optimizer, *op, input_op, parent, child_stats, limit_op, datasource_filters);
		return true;
	}
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		auto &join = op->Cast<LogicalComparisonJoin>();
		// Adding relations of the left side to the current join order optimizer
		bool can_reorder_left = ExtractJoinRelations(optimizer, *op->children[0], filter_operators, op);
		bool can_reorder_right = true;
		// For semi & anti joins, you only reorder relations in the left side of the join.
		// We do not want to reorder a relation A into the right side because then all column bindings A from A will be
		// lost after the semi or anti join

		// We cannot reorder a relation B out of the right side because any filter/join in the right side
		// between a relation B and another RHS relation will be invalid. The semi join will remove
		// all right column bindings,

		// So we treat the right side of left join as its own relation so no relations
		// are pushed into the right side, or taken out of the right side.
		if (join.join_type == JoinType::SEMI || join.join_type == JoinType::ANTI) {
			RelationStats child_stats;
			// optimize the child and copy the stats
			auto child_optimizer = optimizer.CreateChildOptimizer();
			op->children[1] = child_optimizer.Optimize(std::move(op->children[1]), &child_stats);
			AddRelation(*op->children[1], op, child_stats);
			// remember that if a cross product needs to be forced, it cannot be forced
			// across the children of a semi or anti join
			no_cross_product_relations.insert(relations.size() - 1);
			auto right_child_bindings = op->children[1]->GetColumnBindings();
			for (auto &bindings : right_child_bindings) {
				relation_mapping[bindings.table_index] = RelationIndex(relations.size() - 1);
			}
		} else {
			can_reorder_right = ExtractJoinRelations(optimizer, *op->children[1], filter_operators, op);
		}
		return can_reorder_left && can_reorder_right;
	}
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT: {
		bool can_reorder_left = ExtractJoinRelations(optimizer, *op->children[0], filter_operators, op);
		bool can_reorder_right = ExtractJoinRelations(optimizer, *op->children[1], filter_operators, op);
		return can_reorder_left && can_reorder_right;
	}
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN: {
		auto &dummy_scan = op->Cast<LogicalDummyScan>();
		auto stats = RelationStatisticsHelper::ExtractDummyScanStats(dummy_scan, context);
		AddRelation(input_op, parent, stats);
		return true;
	}
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET: {
		// base table scan, add to set of relations.
		// create empty stats for dummy scan or logical expression get
		auto &expression_get = op->Cast<LogicalExpressionGet>();
		auto stats = RelationStatisticsHelper::ExtractExpressionGetStats(expression_get, context);
		AddRelation(input_op, parent, stats);
		return true;
	}
	case LogicalOperatorType::LOGICAL_GET: {
		// TODO: Get stats from a logical GET
		auto &get = op->Cast<LogicalGet>();
		// this is a get that *most likely* has a function (like unnest or json_each).
		// there are new bindings for output of the function, but child bindings also exist, and can
		// be used in joins
		if (!op->children.empty()) {
			RelationStats child_stats;
			AddRelationWithChildren(optimizer, *op, input_op, parent, child_stats, limit_op, datasource_filters);
			return true;
		}
		auto stats = RelationStatisticsHelper::ExtractGetStats(get, context);
		// if there is another logical filter that could not be pushed down into the
		// table scan, apply another selectivity.
		get.SetEstimatedCardinality(stats.cardinality);
		if (!datasource_filters.empty()) {
			stats.cardinality =
			    (idx_t)MaxValue(double(stats.cardinality) * RelationStatisticsHelper::DEFAULT_SELECTIVITY, (double)1);
		}
		ModifyStatsIfLimit(limit_op.get(), stats);
		AddRelation(input_op, parent, stats);
		return true;
	}
	case LogicalOperatorType::LOGICAL_PROJECTION: {
		RelationStats child_stats;
		// optimize the child and copy the stats
		auto child_optimizer = optimizer.CreateChildOptimizer();
		op->children[0] = child_optimizer.Optimize(std::move(op->children[0]), &child_stats);
		auto &proj = op->Cast<LogicalProjection>();
		// Projection can create columns so we need to add them here
		auto proj_stats = RelationStatisticsHelper::ExtractProjectionStats(proj, child_stats);
		proj.SetEstimatedCardinality(proj_stats.cardinality);
		ModifyStatsIfLimit(limit_op.get(), proj_stats);
		AddRelation(input_op, parent, proj_stats);
		return true;
	}
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT: {
		// optimize the child and copy the stats
		auto &empty_result = op->Cast<LogicalEmptyResult>();
		// Projection can create columns so we need to add them here
		auto stats = RelationStatisticsHelper::ExtractEmptyResultStats(empty_result);
		empty_result.SetEstimatedCardinality(stats.cardinality);
		AddRelation(input_op, parent, stats);
		return true;
	}
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE: {
		RelationStats lhs_stats;
		// optimize the lhs child and copy the stats
		auto lhs_optimizer = optimizer.CreateChildOptimizer();
		op->children[0] = lhs_optimizer.Optimize(std::move(op->children[0]), &lhs_stats);
		// optimize the rhs child
		auto rhs_optimizer = optimizer.CreateChildOptimizer();
		auto table_index = op->Cast<LogicalCTE>().table_index;

		auto child_1_card = lhs_stats.stats_initialized ? lhs_stats.cardinality : 0;
		rhs_optimizer.AddMaterializedCTEStats(table_index, std::move(lhs_stats));
		if (op->type == LogicalOperatorType::LOGICAL_RECURSIVE_CTE) {
			rhs_optimizer.recursive_cte_indexes.insert(op->Cast<LogicalCTE>().table_index);
		}
		RelationStats rhs_stats;
		op->children[1] = rhs_optimizer.Optimize(std::move(op->children[1]), &rhs_stats);

		// create the stats for the CTE
		auto child_2_card = rhs_stats.stats_initialized ? rhs_stats.cardinality : 0;

		if (op->type == LogicalOperatorType::LOGICAL_RECURSIVE_CTE) {
			// we cannot really estimate the cardinality of a recursive CTE
			// because we don't know how many times it will be executed
			// we just assume it will be executed 1000 times
			op->SetEstimatedCardinality(child_1_card + child_2_card * 1000);
		} else if (op->type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
			// for a materialized CTE, we just take the cardinality of the right children
			op->SetEstimatedCardinality(child_2_card);
		}

		return false;
	}
	case LogicalOperatorType::LOGICAL_CTE_REF: {
		auto &cte_ref = op->Cast<LogicalCTERef>();
		auto cte_stats = optimizer.GetMaterializedCTEStats(cte_ref.cte_index);
		cte_ref.SetEstimatedCardinality(cte_stats.cardinality);
		AddRelation(input_op, parent, cte_stats);

		auto is_recursive = optimizer.recursive_cte_indexes.find(cte_ref.cte_index);
		if (is_recursive != optimizer.recursive_cte_indexes.end()) {
			return false;
		}
		return true;
	}
	case LogicalOperatorType::LOGICAL_DELIM_JOIN: {
		auto &delim_join = op->Cast<LogicalComparisonJoin>();

		// optimize LHS (duplicate-eliminated) child
		RelationStats lhs_stats;
		auto lhs_optimizer = optimizer.CreateChildOptimizer();
		op->children[0] = lhs_optimizer.Optimize(std::move(op->children[0]), &lhs_stats);

		// create dummy aggregation for the duplicate elimination
		auto dummy_aggr = make_uniq<LogicalAggregate>(TableIndex(DConstants::INVALID_INDEX - 1), TableIndex(),
		                                              vector<unique_ptr<Expression>>());
		dummy_aggr->grouping_sets.emplace_back();
		for (auto &delim_col : delim_join.duplicate_eliminated_columns) {
			dummy_aggr->grouping_sets.back().insert(ProjectionIndex(dummy_aggr->groups.size()));
			dummy_aggr->groups.push_back(delim_col->Copy());
		}
		auto lhs_delim_stats = RelationStatisticsHelper::ExtractAggregationStats(*dummy_aggr, lhs_stats);

		// optimize the other child, which will now have access to the stats
		RelationStats rhs_stats;
		auto rhs_optimizer = optimizer.CreateChildOptimizer();
		rhs_optimizer.AddDelimScanStats(lhs_delim_stats);
		op->children[1] = rhs_optimizer.Optimize(std::move(op->children[1]), rhs_stats);

		RelationStats dj_stats;
		switch (delim_join.join_type) {
		case JoinType::LEFT:
		case JoinType::INNER:
		case JoinType::OUTER:
		case JoinType::SINGLE:
		case JoinType::MARK:
		case JoinType::SEMI:
		case JoinType::ANTI:
			dj_stats = lhs_stats;
			break;
		case JoinType::RIGHT:
		case JoinType::RIGHT_SEMI:
		case JoinType::RIGHT_ANTI:
			dj_stats = rhs_stats;
			break;
		default:
			throw NotImplementedException("Unsupported join type");
		}

		if (delim_join.join_type == JoinType::SEMI || delim_join.join_type == JoinType::ANTI ||
		    delim_join.join_type == JoinType::RIGHT_SEMI || delim_join.join_type == JoinType::RIGHT_ANTI) {
			dj_stats.cardinality =
			    MaxValue<idx_t>(LossyNumericCast<idx_t>(static_cast<double>(dj_stats.cardinality) /
			                                            CardinalityEstimator::DEFAULT_SEMI_ANTI_SELECTIVITY),
			                    1);
		}

		AddAggregateOrWindowRelation(input_op, parent, dj_stats, op->type);

		return false;
	}
	case LogicalOperatorType::LOGICAL_DELIM_GET: {
		// Used to not be possible to reorder these. We added reordering (without stats) before,
		// but ran into terrible join orders (see internal issue #596), so we removed it again
		// We now have proper statistics for DelimGets, and get an even better query plan for #596
		auto delim_scan_stats = optimizer.GetDelimScanStats();
		op->SetEstimatedCardinality(delim_scan_stats.cardinality);
		AddAggregateOrWindowRelation(input_op, parent, delim_scan_stats, op->type);
		return true;
	}
	default:
		return false;
	}
}

bool RelationManager::ExtractBindings(Expression &expression, unordered_set<RelationIndex> &bindings) {
	if (expression.GetExpressionType() == ExpressionType::BOUND_COLUMN_REF) {
		auto &colref = expression.Cast<BoundColumnRefExpression>();
		D_ASSERT(colref.depth == 0);
		D_ASSERT(colref.binding.table_index.IsValid());
		// map the base table index to the relation index used by the JoinOrderOptimizer
		if (expression.GetAlias() == "SUBQUERY" &&
		    relation_mapping.find(colref.binding.table_index) == relation_mapping.end()) {
			// most likely a BoundSubqueryExpression that was created from an uncorrelated subquery
			// Here we return true and don't fill the bindings, the expression can be reordered.
			// A filter will be created using this expression, and pushed back on top of the parent
			// operator during plan reconstruction
			return true;
		}
		if (relation_mapping.find(colref.binding.table_index) != relation_mapping.end()) {
			bindings.insert(relation_mapping[colref.binding.table_index]);
		}
	}
	if (expression.GetExpressionType() == ExpressionType::BOUND_REF) {
		// bound expression
		bindings.clear();
		return false;
	}
	D_ASSERT(expression.GetExpressionType() != ExpressionType::SUBQUERY);
	bool can_reorder = true;
	ExpressionIterator::EnumerateChildren(expression, [&](Expression &expr) {
		if (!ExtractBindings(expr, bindings)) {
			can_reorder = false;
			return;
		}
	});
	return can_reorder;
}

vector<unique_ptr<FilterInfo>> RelationManager::ExtractEdges(LogicalOperator &op,
                                                             vector<reference<LogicalOperator>> &filter_operators,
                                                             JoinRelationSetManager &set_manager) {
	// now that we know we are going to perform join ordering we actually extract the filters, eliminating duplicate
	// filters in the process
	vector<unique_ptr<FilterInfo>> filters_and_bindings;
	expression_set_t filter_set;
	for (auto &filter_op : filter_operators) {
		auto &f_op = filter_op.get();
		if (f_op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
		    f_op.type == LogicalOperatorType::LOGICAL_ASOF_JOIN) {
			auto &join = f_op.Cast<LogicalComparisonJoin>();
			D_ASSERT(join.expressions.empty());
			if (join.join_type == JoinType::SEMI || join.join_type == JoinType::ANTI) {
				auto conjunction_expression = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
				// create a conjunction expression for the semi join.
				// It's possible multiple LHS relations have a condition in
				// this semi join. Suppose we have ((A ⨝ B) ⋉ C). (example in test_4950.test)
				// If the semi join condition has A.x = C.y AND B.x = C.z then we need to prevent a reordering
				// that looks like ((A ⋉ C) ⨝ B)), since all columns from C will be lost after it joins with A,
				// and the condition B.x = C.z will no longer be possible.
				// if we make a conjunction expressions and populate the left set and right set with all
				// the relations from the conditions in the conjunction expression, we can prevent invalid
				// reordering.
				for (auto &cond : join.conditions) {
					if (cond.IsComparison()) {
						auto comparison = make_uniq<BoundComparisonExpression>(
						    cond.GetComparisonType(), cond.GetLHS().Copy(), cond.GetRHS().Copy());
						conjunction_expression->children.push_back(std::move(comparison));
					}
				}

				// create the filter info so all required LHS relations are present when reconstructing the
				// join
				optional_ptr<JoinRelationSet> left_set;
				optional_ptr<JoinRelationSet> right_set;
				optional_ptr<JoinRelationSet> full_set;
				// here we create a left_set that unions all relations from the left side of
				// every expression and a right_set that unions all relations frmo the right side of a
				// every expression (although this should always be 1).
				for (auto &bound_expr : conjunction_expression->children) {
					D_ASSERT(bound_expr->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON);
					auto &comp = bound_expr->Cast<BoundComparisonExpression>();
					unordered_set<RelationIndex> right_bindings, left_bindings;
					ExtractBindings(*comp.right, right_bindings);
					ExtractBindings(*comp.left, left_bindings);

					if (!left_set) {
						left_set = set_manager.GetJoinRelation(left_bindings);
					} else {
						left_set = set_manager.Union(set_manager.GetJoinRelation(left_bindings), *left_set);
					}
					if (!right_set) {
						right_set = set_manager.GetJoinRelation(right_bindings);
					} else {
						right_set = set_manager.Union(set_manager.GetJoinRelation(right_bindings), *right_set);
					}
				}
				full_set = set_manager.Union(*left_set, *right_set);
				D_ASSERT(left_set && left_set->count > 0);
				D_ASSERT(right_set && right_set->count == 1);
				D_ASSERT(full_set && full_set->count > 0);

				// now we push the conjunction expressions
				// In QueryGraphManager::GenerateJoins we extract each condition again and create a standalone join
				// condition.
				auto filter_info = make_uniq<FilterInfo>(std::move(conjunction_expression), *full_set,
				                                         filters_and_bindings.size(), join.join_type);
				filter_info->SetLeftSet(left_set);
				filter_info->SetRightSet(right_set);

				filters_and_bindings.push_back(std::move(filter_info));
			} else {
				// can extract every inner join condition individually.
				for (auto &cond : join.conditions) {
					unique_ptr<Expression> expr;
					bool is_residual = false;

					if (cond.IsComparison()) {
						auto comp_type = cond.GetComparisonType();
						expr =
						    make_uniq<BoundComparisonExpression>(comp_type, cond.GetLHS().Copy(), cond.GetRHS().Copy());
					} else {
						expr = cond.GetJoinExpression().Copy();
						is_residual = true;
					}

					if (filter_set.find(*expr) == filter_set.end()) {
						filter_set.insert(*expr);
						unordered_set<RelationIndex> bindings;
						ExtractBindings(*expr, bindings);
						auto &set = set_manager.GetJoinRelation(bindings);
						auto filter_info =
						    make_uniq<FilterInfo>(std::move(expr), set, filters_and_bindings.size(), join.join_type);
						filter_info->from_residual_predicate = is_residual;
						filters_and_bindings.push_back(std::move(filter_info));
					}
				}
			}

			join.conditions.clear();
		} else {
			vector<unique_ptr<Expression>> leftover_expressions;
			for (auto &expression : f_op.expressions) {
				if (filter_set.find(*expression) == filter_set.end()) {
					filter_set.insert(*expression);
					unordered_set<RelationIndex> bindings;
					ExtractBindings(*expression, bindings);
					if (bindings.empty()) {
						// the filter is on a column that is not in our relational map. (example: limit_rownum)
						// in this case we do not create a FilterInfo for it. (duckdb-internal/#1493)s
						leftover_expressions.push_back(std::move(expression));
						continue;
					}
					auto &set = set_manager.GetJoinRelation(bindings);
					auto filter_info = make_uniq<FilterInfo>(std::move(expression), set, filters_and_bindings.size());
					filters_and_bindings.push_back(std::move(filter_info));
				}
			}
			f_op.expressions = std::move(leftover_expressions);
		}
	}

	return filters_and_bindings;
}

// LCOV_EXCL_START

void RelationManager::PrintRelationStats() {
#ifdef DEBUG
	string to_print;
	for (idx_t i = 0; i < relations.size(); i++) {
		auto &relation = relations.at(i);
		auto &stats = relation->stats;
		D_ASSERT(stats.column_names.size() == stats.column_distinct_count.size());
		for (idx_t i = 0; i < stats.column_names.size(); i++) {
			to_print = stats.column_names.at(i) + " has estimated distinct count " +
			           to_string(stats.column_distinct_count.at(i).distinct_count);
			Printer::Print(to_print);
		}
		to_print = stats.table_name + " has estimated cardinality " + to_string(stats.cardinality);
		to_print += " and relation id " + to_string(i) + "\n";
		Printer::Print(to_print);
	}
#endif
}

// LCOV_EXCL_STOP

} // namespace duckdb
