#include "duckdb/optimizer/join_order/query_graph.hpp"

#include "duckdb/common/printer.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/assert.hpp"

namespace duckdb {

using QueryEdge = QueryGraphEdges::QueryEdge;

// LCOV_EXCL_START
static string QueryEdgeToString(const QueryEdge *info, vector<RelationIndex> prefix) {
	string result = "";
	string source = "[";
	for (idx_t i = 0; i < prefix.size(); i++) {
		source += to_string(prefix[i].index) + (i < prefix.size() - 1 ? ", " : "");
	}
	source += "]";
	for (auto &entry : info->neighbors) {
		result += StringUtil::Format("%s -> %s\n", source.c_str(), entry->neighbor->ToString().c_str());
	}
	for (auto &entry : info->children) {
		vector<RelationIndex> new_prefix = prefix;
		new_prefix.push_back(entry.first);
		result += QueryEdgeToString(entry.second.get(), new_prefix);
	}
	return result;
}

string QueryGraphEdges::ToString() const {
	return QueryEdgeToString(&root, {});
}

void QueryGraphEdges::Print() {
	Printer::Print(ToString());
}
// LCOV_EXCL_STOP

optional_ptr<QueryEdge> QueryGraphEdges::GetQueryEdge(JoinRelationSet &left) {
	D_ASSERT(left.count > 0);
	// find the EdgeInfo corresponding to the left set
	optional_ptr<QueryEdge> info(&root);
	for (idx_t i = 0; i < left.count; i++) {
		auto entry = info.get()->children.find(left.relations[i]);
		if (entry == info.get()->children.end()) {
			// node not found, create it
			auto insert_it = info.get()->children.insert(make_pair(left.relations[i], make_uniq<QueryEdge>()));
			entry = insert_it.first;
		}
		// move to the next node
		info = entry->second;
	}
	return info;
}

// 创建一条边，就是查Trie树，为某节点添加neighbor（表示边的另一端）
// 很明显，这是一个单向边，还需要从另外一个方向再调用一次，构成双向边。
void QueryGraphEdges::CreateEdge(JoinRelationSet &left, JoinRelationSet &right, optional_ptr<FilterInfo> filter_info) {
	D_ASSERT(left.count > 0 && right.count > 0);
	// 查找与左侧集合对应的 EdgeInfo
	auto info = GetQueryEdge(left);
	// 现在将边插入到右侧关系中，如果该边尚不存在的话
	for (idx_t i = 0; i < info->neighbors.size(); i++) {
		if (info->neighbors[i]->neighbor == &right) {
			if (filter_info) {
				// 邻接节点已存在，只需添加过滤条件（如果有的话）
				info->neighbors[i]->filters.push_back(filter_info);
			}
			return;
		}
	}
	// 邻接节点不存在，创建它
	auto n = make_uniq<NeighborInfo>(&right);
	// 如果该边表示笛卡尔积（cross product），则 filter_info 为 null。
	// 因此，判断一条边是否用于笛卡尔积的最简单方法就是检查其过滤条件是否为空。
	if (info && filter_info) {
		n->filters.push_back(filter_info);
	}
	info->neighbors.push_back(std::move(n));
}

void QueryGraphEdges::EnumerateNeighborsDFS(JoinRelationSet &node, reference<QueryEdge> info, idx_t index,
                                            const std::function<bool(NeighborInfo &)> &callback) const {
	for (auto &neighbor : info.get().neighbors) {
		if (callback(*neighbor)) {
			return;
		}
	}

	for (idx_t node_index = index; node_index < node.count; ++node_index) {
		auto iter = info.get().children.find(node.relations[node_index]);
		if (iter != info.get().children.end()) {
			reference<QueryEdge> new_info = *iter->second;
			EnumerateNeighborsDFS(node, new_info, node_index + 1, callback);
		}
	}
}

void QueryGraphEdges::EnumerateNeighbors(JoinRelationSet &node,
                                         const std::function<bool(NeighborInfo &)> &callback) const {
	for (idx_t j = 0; j < node.count; j++) {
		auto iter = root.children.find(node.relations[j]);
		if (iter != root.children.end()) {
			reference<QueryEdge> new_info = *iter->second;
			EnumerateNeighborsDFS(node, new_info, j + 1, callback);
		}
	}
}

//! Returns true if a JoinRelationSet is banned by the list of exclusion_set, false otherwise
static bool JoinRelationSetIsExcluded(optional_ptr<JoinRelationSet> node, unordered_set<RelationIndex> &exclusion_set) {
	return exclusion_set.find(node->relations[0]) != exclusion_set.end();
}

const vector<RelationIndex> QueryGraphEdges::GetNeighbors(JoinRelationSet &node,
                                                          unordered_set<RelationIndex> &exclusion_set) const {
	unordered_set<RelationIndex> result;
	EnumerateNeighbors(node, [&](NeighborInfo &info) -> bool {
		if (!JoinRelationSetIsExcluded(info.neighbor, exclusion_set)) {
			// add the smallest node of the neighbor to the set
			result.insert(info.neighbor->relations[0]);
		}
		return false;
	});
	vector<RelationIndex> neighbors;
	neighbors.insert(neighbors.end(), result.begin(), result.end());
	return neighbors;
}

const vector<reference<NeighborInfo>> QueryGraphEdges::GetConnections(JoinRelationSet &node,
                                                                      JoinRelationSet &other) const {
	vector<reference<NeighborInfo>> connections;
	EnumerateNeighbors(node, [&](NeighborInfo &info) -> bool {
		if (JoinRelationSet::IsSubset(other, *info.neighbor)) {
			connections.push_back(info);
		}
		return false;
	});
	return connections;
}

} // namespace duckdb
