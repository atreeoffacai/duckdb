//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/subquery/flatten_dependent_join.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/unordered_map.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

//! FlattenDependentJoins 类负责将依赖连接推送到计划中，以创建扁平化的子查询
struct FlattenDependentJoins {
	FlattenDependentJoins(Binder &binder, const CorrelatedColumns &correlated, bool perform_delim = true,
	                      bool any_join = false, optional_ptr<FlattenDependentJoins> parent = nullptr);

	static unique_ptr<LogicalOperator> DecorrelateIndependent(Binder &binder, unique_ptr<LogicalOperator> plan);

	unique_ptr<LogicalOperator> Decorrelate(unique_ptr<LogicalOperator> plan, bool parent_propagate_null_values = true,
	                                        idx_t lateral_depth = 0);

private:
	//! 检测哪些逻辑操作符具有它们所依赖的相关表达式，并填充 has_correlated_expressions 映射。
	bool DetectCorrelatedExpressions(LogicalOperator &op, bool lateral = false, idx_t lateral_depth = 0,
	                                 bool parent_is_dependent_join = false);

	//! 通过将整个逻辑操作符子树添加到 has_correlated_expressions 映射中，将其标记为相关。
	bool MarkSubtreeCorrelated(LogicalOperator &op, TableIndex cte_index);

	//! 将依赖连接向下推送到逻辑操作符中
	unique_ptr<LogicalOperator> PushDownDependentJoin(unique_ptr<LogicalOperator> plan,
	                                                  bool propagates_null_values = true, idx_t lateral_depth = 0);

public:
	Binder &binder;
	ColumnBinding base_binding;
	idx_t delim_offset;
	idx_t data_offset;
	reference_map_t<LogicalOperator, bool> has_correlated_expressions;
	column_binding_map_t<idx_t> correlated_map;
	column_binding_map_t<idx_t> replacement_map;
	const CorrelatedColumns &correlated_columns;
	vector<LogicalType> delim_types;

	bool perform_delim;
	bool any_join;
	optional_ptr<FlattenDependentJoins> parent;

private:
	unique_ptr<LogicalOperator> PushDownDependentJoinInternal(unique_ptr<LogicalOperator> plan,
	                                                          bool &parent_propagate_null_values, idx_t lateral_depth);
};

} // namespace duckdb
