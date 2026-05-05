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
// 面向已经确定有单个子节点的算子
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

// 在LogicalOperatorType层面指明哪一些操作符是不能被重排序的，面向有双子节点的算子
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
	// 这里你有一个对表中单列的过滤器。返回被过滤列的绑定，
	// 以便过滤器估计器知道要拉取哪个HLL计数。
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

// 从 当前 op 开始，沿着单子树往下走，直到遇到一个LOGICAL_COMPARISON_JOIN算子或者没有子节点的算子为止；
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

// xm comment: 函数的返回值代表能否重排，true代表可以重排，false代表不可以重排；
// 更详细一点，现在是DPHyp的“准备阶段”，返回true表示一切顺利，返回 false 表示遇到了问题。
// filter_operators 用来保存在此过程中收集到的带 condition 的算子：LOGICAL_FILTER，可重排的 Join。
// 假如这个函数形成了多个relation，那么这些relation在初始计划树中都通过哪些operator相连接？一定是Join
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
				datasource_filters.push_back(*op); // 只是为了修正基数，DP是不会把这个filter算子当成一个relation来处理的；可是filter算子下面那个算子可能会被当成一个relation
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
	if (non_reorderable_operation) { // non_reorderable_operation 特指二元操作符
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
		AddRelation(input_op, parent, combined_stats);// 把这个input_op看成一个relation，包括前面穿透的算子
		return true;
	}

	switch (op->type) {
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: { // OperatorNeedsRelation
		// 分组聚合也是不可以重排的，要当成一个relation来处理，但是其基数估计和“non_reorderable_operation”系列不同，所以要单独处理
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
		AddAggregateOrWindowRelation(input_op, parent, operator_stats, op->type); // 特殊的AddRelation
		return true;
	}
	case LogicalOperatorType::LOGICAL_WINDOW: { // OperatorNeedsRelation
		// 类似于前面的分组聚合
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
		AddAggregateOrWindowRelation(input_op, parent, operator_stats, op->type); // 特殊的AddRelation
		return true;
	}
	case LogicalOperatorType::LOGICAL_UNNEST: { // OperatorNeedsRelation
		// 那unnest及其子树当成一个relation
		// 不过包装成了函数AddRelationWithChildren(),包装的原因是，LOGICAL_GET也用了这个逻辑
		RelationStats child_stats;
		AddRelationWithChildren(optimizer, *op, input_op, parent, child_stats, limit_op, datasource_filters);
		return true;
	}
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: { // 可重排的比较连接
		auto &join = op->Cast<LogicalComparisonJoin>();
		// 将左侧的关系添加到当前的连接顺序优化器中
		bool can_reorder_left = ExtractJoinRelations(optimizer, *op->children[0], filter_operators, op);
		bool can_reorder_right = true;
		// 对于半连接（semi join）和反连接（anti join），你只能对连接左侧的关系进行重排序。
		// 我们不希望将某个关系 A 重排到右侧，因为那样的话，
		// 所有来自 A 的列绑定（column bindings）在半连接或反连接之后都会丢失。

		// 同样，我们也不能将某个关系 B 从右侧移出，
		// 理由如下：半连接会移除所有右侧的列绑定，一旦关系 B 从右侧移出，右侧中关系 B 与其他右侧关系之间的任何过滤条件或连接条件都将变得无效（那些列消失了）。

		// 因此，我们将左连接的右侧视为一个独立的关系，
		// 这样就不会有关系被推入右侧，也不会有关系被从右侧移出。（重点）
		// （xm: 把右边打包成一个relation，就可以保证：右边不会有关系出去，也不会有关系进来。但是光凭这来保证join Reorder的正确性还不够，
		// 还需要构建对应的超边来约束，详见ExtractEdges)
		if (join.join_type == JoinType::SEMI || join.join_type == JoinType::ANTI) {
			RelationStats child_stats;
			// optimize the child and copy the stats
			auto child_optimizer = optimizer.CreateChildOptimizer();
			op->children[1] = child_optimizer.Optimize(std::move(op->children[1]), &child_stats);
			AddRelation(*op->children[1], op, child_stats); // 优化右侧，并把整个右侧看成一个relation
			// 请注意，如果需要强制执行笛卡尔积（cross product），
			// 则不能在半连接（semi join）或反连接（anti join）的子节点之间强制执行。
			no_cross_product_relations.insert(relations.size() - 1);
			auto right_child_bindings = op->children[1]->GetColumnBindings(); // AddRelation已经为基表建立了映射，但是这里为了防止有新的生成列不在其中，又根据列进行了映射
			for (auto &bindings : right_child_bindings) {
				relation_mapping[bindings.table_index] = RelationIndex(relations.size() - 1); 
			}
		} else { // 普通连接
			can_reorder_right = ExtractJoinRelations(optimizer, *op->children[1], filter_operators, op);
		}
		return can_reorder_left && can_reorder_right;
	}
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT: {
		bool can_reorder_left = ExtractJoinRelations(optimizer, *op->children[0], filter_operators, op);
		bool can_reorder_right = ExtractJoinRelations(optimizer, *op->children[1], filter_operators, op);
		return can_reorder_left && can_reorder_right;
	}
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN: { // 假表，但是基数被设置为1，要当成一个relation来处理
		auto &dummy_scan = op->Cast<LogicalDummyScan>();
		auto stats = RelationStatisticsHelper::ExtractDummyScanStats(dummy_scan, context);
		AddRelation(input_op, parent, stats);
		return true;
	}
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET: {
		// 基表扫描，将其加入关系集合中。
		// 为虚拟扫描（dummy scan）或逻辑表达式获取创建空的统计信息。
		auto &expression_get = op->Cast<LogicalExpressionGet>();
		auto stats = RelationStatisticsHelper::ExtractExpressionGetStats(expression_get, context);
		AddRelation(input_op, parent, stats);
		return true;
	}
	case LogicalOperatorType::LOGICAL_GET: {
		// TODO: Get stats from a logical GET
		auto &get = op->Cast<LogicalGet>();
		// 这是一个“Get”操作，*很可能*包含一个函数（例如 unnest 或 json_each）。
		// 该函数的输出会引入新的绑定（bindings），但子节点的绑定也依然存在，
		// 并可用于连接（joins）操作。
		if (!op->children.empty()) { // Get竟然可以有子节点，把其当成一个relation来处理
			RelationStats child_stats;
			AddRelationWithChildren(optimizer, *op, input_op, parent, child_stats, limit_op, datasource_filters);
			return true;
		}
		auto stats = RelationStatisticsHelper::ExtractGetStats(get, context);
		// 如果还存在另一个无法下推到表扫描中的逻辑过滤条件，
		// 则应用另一个选择率（selectivity）。
		get.SetEstimatedCardinality(stats.cardinality);
		if (!datasource_filters.empty()) {
			stats.cardinality =
			    (idx_t)MaxValue(double(stats.cardinality) * RelationStatisticsHelper::DEFAULT_SELECTIVITY, (double)1);
		}
		ModifyStatsIfLimit(limit_op.get(), stats);
		AddRelation(input_op, parent, stats);
		return true;
	}
	case LogicalOperatorType::LOGICAL_PROJECTION: { // OperatorNeedsRelation
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
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT: { // 行数为0的数据源，基数为0，要当成一个relation来处理
		// optimize the child and copy the stats
		auto &empty_result = op->Cast<LogicalEmptyResult>();
		// Projection can create columns so we need to add them here
		auto stats = RelationStatisticsHelper::ExtractEmptyResultStats(empty_result);
		empty_result.SetEstimatedCardinality(stats.cardinality);
		AddRelation(input_op, parent, stats);
		return true;
	}
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE: // 物化CTE
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE: { // 递归CTE，这两个是二元算子
		RelationStats lhs_stats;
		// optimize the lhs child and copy the stats 优化左子树，一般这是CTE的定义部分
		auto lhs_optimizer = optimizer.CreateChildOptimizer();
		op->children[0] = lhs_optimizer.Optimize(std::move(op->children[0]), &lhs_stats);
		// optimize the rhs child 优化右子树，一般这是CTE的引用部分
		auto rhs_optimizer = optimizer.CreateChildOptimizer();
		auto table_index = op->Cast<LogicalCTE>().table_index;

		auto child_1_card = lhs_stats.stats_initialized ? lhs_stats.cardinality : 0;
		rhs_optimizer.AddMaterializedCTEStats(table_index, std::move(lhs_stats)); // 右边肯定引用了左边定义的CTE，把左边的统计信息给右边使用
		if (op->type == LogicalOperatorType::LOGICAL_RECURSIVE_CTE) {
			rhs_optimizer.recursive_cte_indexes.insert(op->Cast<LogicalCTE>().table_index);
		}
		RelationStats rhs_stats;
		op->children[1] = rhs_optimizer.Optimize(std::move(op->children[1]), &rhs_stats); // 优化右子树

		// create the stats for the CTE
		auto child_2_card = rhs_stats.stats_initialized ? rhs_stats.cardinality : 0;

		if (op->type == LogicalOperatorType::LOGICAL_RECURSIVE_CTE) {
		// 我们实际上无法准确估算递归CTE的基数（cardinality），
		// 因为我们不知道它会被执行多少次，
		// 因此我们简单地假设它将被执行1000次。
			op->SetEstimatedCardinality(child_1_card + child_2_card * 1000);
		} else if (op->type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
			// for a materialized CTE, we just take the cardinality of the right children
			op->SetEstimatedCardinality(child_2_card);
		}

		return false;
	}
	case LogicalOperatorType::LOGICAL_CTE_REF: { // 应用CTE的地方
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
	case LogicalOperatorType::LOGICAL_DELIM_JOIN: { // 单独优化左侧，然后单独优化右侧，然后本层DPHpy取消......(return false)
		auto &delim_join = op->Cast<LogicalComparisonJoin>();

		// optimize LHS (duplicate-eliminated) child 单独优化左子树
		RelationStats lhs_stats;
		auto lhs_optimizer = optimizer.CreateChildOptimizer();
		op->children[0] = lhs_optimizer.Optimize(std::move(op->children[0]), &lhs_stats);

		// create dummy aggregation for the duplicate elimination
		// 这是利用分区聚合来估算去重后的基数
		auto dummy_aggr = make_uniq<LogicalAggregate>(TableIndex(DConstants::INVALID_INDEX - 1), TableIndex(),
		                                              vector<unique_ptr<Expression>>());
		dummy_aggr->grouping_sets.emplace_back();
		for (auto &delim_col : delim_join.duplicate_eliminated_columns) {
			dummy_aggr->grouping_sets.back().insert(ProjectionIndex(dummy_aggr->groups.size()));
			dummy_aggr->groups.push_back(delim_col->Copy());
		}
		auto lhs_delim_stats = RelationStatisticsHelper::ExtractAggregationStats(*dummy_aggr, lhs_stats);

		// optimize the other child, which will now have access to the stats
		// 现在优化右子树
		RelationStats rhs_stats;
		auto rhs_optimizer = optimizer.CreateChildOptimizer();
		// 【关键！】把假聚合算出来的统计情报交接给右侧优化器
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
	case LogicalOperatorType::LOGICAL_DELIM_GET: { // 从 input_op 开始打包
	// 以前无法对这些进行重排序。我们之前曾添加过重排序功能（但没有统计信息），
	// 却导致了非常糟糕的连接顺序（参见内部问题 #596），因此又移除了该功能。
	// 现在我们为 DelimGets 提供了准确的统计信息，从而为问题 #596 生成了更优的查询计划。
		auto delim_scan_stats = optimizer.GetDelimScanStats();
		op->SetEstimatedCardinality(delim_scan_stats.cardinality);
		AddAggregateOrWindowRelation(input_op, parent, delim_scan_stats, op->type);
		return true;
	}
	default: // 典型代表 sample，可能是考虑到“随机”的特殊性
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

// 已知filter_operators要么来自filter算子，要么来自可重排的Join
// 这里需要注意的是：
// 对于filter算子，该函数会把该filter算子中的条件全部移除（方式是使用move转交所有权），也就是说，filter算子位置不变，但是是个空算子。
// 对于搬空之后的filter，在生成物理算子的时候，会被删除。
vector<unique_ptr<FilterInfo>> RelationManager::ExtractEdges(LogicalOperator &op,
                                                             vector<reference<LogicalOperator>> &filter_operators,
                                                             JoinRelationSetManager &set_manager) {
	// 既然现在已经确定要执行连接顺序优化（join ordering），我们就真正地提取过滤条件，
	// 并在此过程中消除重复的过滤条件。
	vector<unique_ptr<FilterInfo>> filters_and_bindings;
	expression_set_t filter_set;
	for (auto &filter_op : filter_operators) { // 函数中的主循环，说明关注的过滤条件都在这里
		auto &f_op = filter_op.get();
		if (f_op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
		    f_op.type == LogicalOperatorType::LOGICAL_ASOF_JOIN) {
			auto &join = f_op.Cast<LogicalComparisonJoin>();
			D_ASSERT(join.expressions.empty());
			if (join.join_type == JoinType::SEMI || join.join_type == JoinType::ANTI) {
				auto conjunction_expression = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
				// 为半连接（semi join）创建一个合取（conjunction）表达式。
				// 在这个半连接中，可能存在多个左表（LHS）关系参与条件判断。
				// 例如，在查询 ((A ⨝ B) ⋉ C) 中（参见 test_4950.test 示例），
				// 如果半连接的条件是 A.x = C.y AND B.x = C.z，
				// 那么我们必须防止重排序成类似 ((A ⋉ C) ⨝ B) 的形式，
				// 因为在 A 与 C 进行半连接后，C 的所有列都会丢失，
				// 导致条件 B.x = C.z 无法再被满足。
				// 如果我们构造一个合取表达式，并将其中所有条件涉及的关系
				// 分别填入左集合（left set）和右集合（right set），
				// 就可以有效阻止这类非法的重排序。
				for (auto &cond : join.conditions) { // 即 通过锁定谓词来限制重排（和创建超边限制重排的思想是一样的）
					if (cond.IsComparison()) {
						auto comparison = make_uniq<BoundComparisonExpression>(
						    cond.GetComparisonType(), cond.GetLHS().Copy(), cond.GetRHS().Copy());
						conjunction_expression->children.push_back(std::move(comparison));
					}
				}

				// 创建过滤信息，以确保在重建连接（join）时所有必需的左表（LHS）关系都存在。
				optional_ptr<JoinRelationSet> left_set;
				optional_ptr<JoinRelationSet> right_set;
				optional_ptr<JoinRelationSet> full_set;
				// 在这里，我们创建一个 left_set，它合并了每个表达式左侧的所有关系；
				// 同时创建一个 right_set，它合并了每个表达式右侧的所有关系（尽管这通常应该只包含一个关系）。
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

				// 现在我们将这些合取（conjunction）表达式压入。
				// 在 QueryGraphManager::GenerateJoins 中，我们会再次逐个提取每个条件，
				// 并创建独立的连接（join）条件。
				auto filter_info = make_uniq<FilterInfo>(std::move(conjunction_expression), *full_set,
				                                         filters_and_bindings.size(), join.join_type);
				filter_info->SetLeftSet(left_set);
				filter_info->SetRightSet(right_set);

				filters_and_bindings.push_back(std::move(filter_info));
			} else {
				// 可以单独提取每一个内连接（inner join）条件。
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

			join.conditions.clear(); // 上面不是“掏空”，而是copy，所以这还需要clear一下；
		} else { // 这里对应的是 filter算子，注意这里的JoinType默认是Inner
			vector<unique_ptr<Expression>> leftover_expressions; // leftover: 剩下的
			for (auto &expression : f_op.expressions) {
				if (filter_set.find(*expression) == filter_set.end()) { // 这两行是去重逻辑
					filter_set.insert(*expression); // 标记已经处理过的过滤条件，避免重复处理
					unordered_set<RelationIndex> bindings;
					ExtractBindings(*expression, bindings); // 提取表
					if (bindings.empty()) {
						// 该过滤条件作用于一个不在我们关系映射中的列上（例如：limit_rownum）。
						// 在这种情况下，我们不会为其创建 FilterInfo。（参见 duckdb-internal/#1493）
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
