#include "duckdb/optimizer/filter_pushdown.hpp"
#include "duckdb/planner/operator/logical_empty_result.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"

namespace duckdb {

using Filter = FilterPushdown::Filter;

// 算子filters->组合器combiner->FilterPushdown的filters
unique_ptr<LogicalOperator> FilterPushdown::PushdownFilter(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_FILTER);
	auto &filter = op->Cast<LogicalFilter>();
	if (filter.HasProjectionMap()) { // filter算子的扩展，自带projection，不支持下推
		return FinishPushdown(std::move(op));
	}
	// filter: 收集过滤器并将过滤器从操作集合中移除 xm：算子filters->组合器combiner
	for (auto &expression : filter.expressions) {
		if (AddFilter(std::move(expression)) == FilterResult::UNSATISFIABLE) {
			// 过滤器静态求值为false，剥离（移除）整棵树
			return make_uniq<LogicalEmptyResult>(std::move(op));
		}
	}
	GenerateFilters(); // 组合器combiner->FilterPushdown的filters
	return Rewrite(std::move(filter.children[0])); // 删除了当前filter算子
}

} // namespace duckdb
