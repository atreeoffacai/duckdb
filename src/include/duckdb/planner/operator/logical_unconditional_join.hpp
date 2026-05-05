//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/operator/logical_unconditional_join.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

//! LogicalUnconditionalJoin 表示两个关系之间的连接，
//! 其中连接条件是隐式的（笛卡尔积、位置等）
class LogicalUnconditionalJoin : public LogicalOperator {
public: // 给 LogicalUnconditionalJoin 这个类绑定一个全局唯一的标签，名字叫 TYPE。这个标签的值是 LOGICAL_INVALID
	static constexpr const LogicalOperatorType TYPE = LogicalOperatorType::LOGICAL_INVALID;

public:
	explicit LogicalUnconditionalJoin(LogicalOperatorType logical_type) : LogicalOperator(logical_type) {};

public:
	LogicalUnconditionalJoin(LogicalOperatorType logical_type, unique_ptr<LogicalOperator> left,
	                         unique_ptr<LogicalOperator> right);

public:
	vector<ColumnBinding> GetColumnBindings() override;

protected:
	void ResolveTypes() override;
};
} // namespace duckdb
