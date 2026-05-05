//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/join_order/join_relation.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/optimizer/join_order/relation_index.hpp"

namespace duckdb {

//! Set of relations, used in the join graph.
struct JoinRelationSet {
	JoinRelationSet(unsafe_unique_array<RelationIndex> relations, idx_t count)
	    : relations(std::move(relations)), count(count) {
	}

	string ToString() const;

	unsafe_unique_array<RelationIndex> relations;
	idx_t count;

	static bool IsSubset(JoinRelationSet &super, JoinRelationSet &sub);
};

//! JoinRelationTree 是一个结构，用于保存所有已创建的 JoinRelationSet 对象，并允许对它们进行快速查找

class JoinRelationSetManager {
public:
	//! 包含一个带有 JoinRelationSet 和子关系的节点
	// FIXME: 此结构效率低下，可以使用位图（bitmap）进行查找（待办：性能分析）
	struct JoinRelationTreeNode {
		unique_ptr<JoinRelationSet> relation;
		unordered_map<RelationIndex, unique_ptr<JoinRelationTreeNode>> children;
	};

public:
	//! 从具有给定索引的单个节点创建或获取一个 JoinRelationSet
	JoinRelationSet &GetJoinRelation(RelationIndex index);
	//! 从一组关系绑定创建或获取一个 JoinRelationSet
	JoinRelationSet &GetJoinRelation(const unordered_set<RelationIndex> &bindings);
	//! 从关系列表（已排序、无重复！）创建或获取一个 JoinRelationSet
	JoinRelationSet &GetJoinRelation(unsafe_unique_array<RelationIndex> relations, idx_t count);
	//! 合并两组关系并创建一个新的关系集合
	JoinRelationSet &Union(JoinRelationSet &left, JoinRelationSet &right);
	// //! 创建左集合与右集合的差集（即左集合中不在右集合中的所有元素）
	// JoinRelationSet *Difference(JoinRelationSet *left, JoinRelationSet *right);
	string ToString() const;
	void Print();

private:
	JoinRelationTreeNode root;
};

} // namespace duckdb
