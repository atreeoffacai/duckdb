//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/enums/join_type.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/constants.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Join Types
//===--------------------------------------------------------------------===//
enum class JoinType : uint8_t {
	INVALID = 0, // 无效的连接类型
	LEFT = 1,    // 左连接
	RIGHT = 2,   // 右连接
	INNER = 3,   // 内连接
	OUTER = 4,   // 外连接
	SEMI = 5,    // 左半连接仅在左侧行有连接伙伴时返回该行，无重复
	ANTI = 6,    // 左反连接仅在左侧行没有连接伙伴时返回该行，无重复
	MARK = 7,    // 标记连接返回一个标记，指示是否存在连接伙伴（true）或不存在连接伙伴（false）
	SINGLE = 8,  // 单一连接类似于左外连接，但每条左侧记录最多返回一个连接伙伴（如果没有找到伙伴则返回 NULL）
	RIGHT_SEMI = 9, // 右半连接由优化器创建，当半连接的子节点需要交换时，以便构建侧可以是较小的表
	RIGHT_ANTI = 10 // 右反连接由优化器创建，当反连接的子节点需要交换时，以便构建侧可以是较小的表
};

//! True if join is left or full outer join
bool IsLeftOuterJoin(JoinType type);

//! True if join is rght or full outer join
bool IsRightOuterJoin(JoinType type);

//! Whether the build side is propagated out of the join
bool PropagatesBuildSide(JoinType type);

//! Whether the JoinType has an inverse
bool HasInverseJoinType(JoinType type);

//! Gets the inverse JoinType, e.g., LEFT -> RIGHT
JoinType InverseJoinType(JoinType type);

// **DEPRECATED**: Use EnumUtil directly instead.
string JoinTypeToString(JoinType type);

} // namespace duckdb
