#include "duckdb/optimizer/filter_pushdown.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_empty_result.hpp"
#include "duckdb/planner/operator/logical_join.hpp"

namespace duckdb {

using Filter = FilterPushdown::Filter;

static unique_ptr<Expression> ReplaceGroupBindings(LogicalAggregate &aggr, unique_ptr<Expression> root_expr) {
	ExpressionIterator::VisitExpressionMutable<BoundColumnRefExpression>(
	    root_expr, [&](BoundColumnRefExpression &colref, unique_ptr<Expression> &expr) {
		    D_ASSERT(colref.depth == 0);
		    // replace the binding with a copy to the expression at the referenced index
		    expr = aggr.GetExpression(colref.binding).Copy();
	    });
	return root_expr;
}

void FilterPushdown::ExtractFilterBindings(const Expression &expr, vector<ColumnBinding> &bindings) {
	ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
	    expr, [&](const BoundColumnRefExpression &colref) { bindings.push_back(colref.binding); });
}

// xm: 先生成一个对子树的谓词下推器，把能下推的谓词给这个新的“谓词下推器”，不能下推的谓词就留在这个op上面形成一个filter；
// 只要谓词相关列，在每一个分组列集合中都能找到，就可以往下推；
// 由于分组聚合会导致schema的变化，所以等价类需要重新推导
unique_ptr<LogicalOperator> FilterPushdown::PushdownAggregate(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY);
	auto &aggr = op->Cast<LogicalAggregate>();

	// pushdown into AGGREGATE and GROUP BY
	// we cannot push expressions that refer to the aggregate
	FilterPushdown child_pushdown(optimizer, convert_mark_joins);
	for (idx_t i = 0; i < filters.size(); i++) {
		auto &f = *filters[i];
		if (f.bindings.find(aggr.aggregate_index) != f.bindings.end()) {
			// filter on aggregate: cannot pushdown
			continue;
		}
		if (f.bindings.find(aggr.groupings_index) != f.bindings.end()) {
			// filter on GROUPINGS function: cannot pushdown
			continue;
		}
		// no aggregate! we are filtering on a group
		// we can only push this down if the filter is in all grouping sets
		if (aggr.groups.empty()) { // xm: 整个表作为一组
			// empty group - we cannot pushdown the filter
			continue;
		}

		vector<ColumnBinding> bindings;
		ExtractFilterBindings(*f.filter, bindings);
		if (bindings.empty()) { // xm：1=1 这种没有绑定的过滤器 不可以下推
			// we can never push down empty grouping sets
			continue;
		}

		bool can_pushdown_filter = true;
		for (auto &grp : aggr.grouping_sets) {
			// check for each of the grouping sets if they contain all groups
			// 条件中涉及的每一个列，要包含在每一个分组列中，才能下推，比如 rollup(a,b)的分组列集合是(a,b) (a) ().谓词中的列必须同时包含在(a,b) (a) ()中才能下推.（显然涉及到rollup就不能下推）
			for (auto &binding : bindings) {
				if (grp.find(binding.column_index) == grp.end()) {
					can_pushdown_filter = false;
					break;
				}
			}
			if (!can_pushdown_filter) {
				break;
			}
		}
		if (!can_pushdown_filter) {
			continue;
		}
		// no aggregate! we can push this down
		// rewrite any group bindings within the filter
		f.filter = ReplaceGroupBindings(aggr, std::move(f.filter)); // 分组聚合会改变列的排布，下推后需要重新定位
		// add the filter to the child node
		if (child_pushdown.AddFilter(std::move(f.filter)) == FilterResult::UNSATISFIABLE) {
			// filter statically evaluates to false, strip tree
			return make_uniq<LogicalEmptyResult>(std::move(op));
		}
		// erase the filter from here
		filters.erase_at(i);
		i--;
	}
	child_pushdown.GenerateFilters();

	op->children[0] = child_pushdown.Rewrite(std::move(op->children[0]));
	return FinishPushdown(std::move(op));
}

} // namespace duckdb
