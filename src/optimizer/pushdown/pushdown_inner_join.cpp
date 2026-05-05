#include "duckdb/optimizer/filter_pushdown.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_empty_result.hpp"

namespace duckdb {

using Filter = FilterPushdown::Filter;

// xm: 对于内连接，收集连接条件，然后变成 笛卡尔积，去按照笛卡尔积的方式进行谓词下推
unique_ptr<LogicalOperator> FilterPushdown::PushdownInnerJoin(unique_ptr<LogicalOperator> op,
                                                              unordered_set<TableIndex> &left_bindings,
                                                              unordered_set<TableIndex> &right_bindings) {
	auto &join = op->Cast<LogicalJoin>();
	D_ASSERT(join.join_type == JoinType::INNER);
	if (op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		op = PushFiltersIntoDelimJoin(std::move(op));
		return FinishPushdown(std::move(op));
	}
	// 内连接：收集内连接的所有条件并添加到过滤器列表中
	if (op->type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		auto &any_join = join.Cast<LogicalAnyJoin>();
		// any join: only one filter to add
		if (AddFilter(std::move(any_join.condition)) == FilterResult::UNSATISFIABLE) {
			// filter statically evaluates to false, strip tree
			return make_uniq<LogicalEmptyResult>(std::move(op));
		}
	} else {
		// comparison join
		D_ASSERT(op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN);
		auto &comp_join = join.Cast<LogicalComparisonJoin>();
		// turn the conditions into filters
		for (auto &i : comp_join.conditions) {
			auto condition = JoinCondition::CreateExpression(std::move(i));
			if (AddFilter(std::move(condition)) == FilterResult::UNSATISFIABLE) {
				// filter statically evaluates to false, strip tree
				return make_uniq<LogicalEmptyResult>(std::move(op));
			}
		}
	}
	GenerateFilters();

	// turn the inner join into a cross product
	auto cross_product = make_uniq<LogicalCrossProduct>(std::move(op->children[0]), std::move(op->children[1]));

	// preserve the estimated cardinality of the operator
	if (op->has_estimated_cardinality) {
		cross_product->SetEstimatedCardinality(op->estimated_cardinality);
	}

	// then push down cross product
	return PushdownCrossProduct(std::move(cross_product));
}

} // namespace duckdb
