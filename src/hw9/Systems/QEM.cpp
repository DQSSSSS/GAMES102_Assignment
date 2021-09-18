#include "QEM.h"

#include <functional>

#include <Eigen/core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <spdlog/spdlog.h>

using namespace Ubpa;

void getPlane(pointf3 A, pointf3 B, pointf3 C, 
	float& a, float& b, float& c, float& d) {
	// a*x + b*y + c*z + d = 0

	valf3 N = (A - B).cross(C - B); // N_x*(x-A_x) + N_y*(y-A_y) + N_z*(y-A_z) = 0
	a = N[0];
	b = N[1];
	c = N[2];
	d = -N[0] * A[0] - N[1] * A[1] - N[2] * A[2];
	auto len = std::sqrt(a * a + b * b + c * c);
	a /= len;
	b /= len;
	c /= len;
	d /= len;
}

std::pair<float, pointf3> getCostAndV(const Eigen::Matrix4f& Q, const pointf3& v1p, const pointf3& v2p) {
	// quadric cost, \bar v
	Eigen::Matrix4f A;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			if (i < 3)
				A(i, j) = Q(i, j);
			else if (j < 3)
				A(i, j) = 0;
			else
				A(i, j) = 1;
		}
	}

	Eigen::Matrix4f inverse;
	bool invertible;
	float determinant;
	A.computeInverseAndDetWithCheck(inverse, determinant, invertible);

	if (invertible) {
		Eigen::Vector4f b;
		for (int i = 0; i < 4; i++) {
			if (i < 3) b(i) = 0;
			else b(i) = 1;
		}
		Eigen::Vector4f v_bar = inverse * b;
		float error = v_bar.transpose() * Q * v_bar;
		pointf3 v_bar_ans;
		for (int i = 0; i < 3; i++) v_bar_ans[i] = v_bar[i] / v_bar[3];
		return std::make_pair(error, v_bar_ans);
	}
	else {
		pointf3 v_bar_ans = (v1p.as<valf3>() + v2p.as<valf3>()) / 2;
		Eigen::Vector4f v_bar;
		for (int i = 0; i < 3; i++) v_bar(i) = v_bar_ans[i];
		v_bar(3) = 1;
		float error = v_bar.transpose() * Q * v_bar;
		return std::make_pair(error, v_bar_ans);
	}
}


void QuadricSimplification::contract() {
	assert(del_number + 1 < checkpoints.size());
	int p = checkpoints[del_number + 1];
	while (worklist_p < p - 1) {
		worklist_p++;
		int tri_id = worklist[worklist_p];
		triangles[tri_id].go();
	}
	del_number++;
}

void QuadricSimplification::contractInv() {
	assert(del_number > 0);
	int p = checkpoints[del_number - 1];
	while (worklist_p >= p) {
		int tri_id = worklist[worklist_p];
		triangles[tri_id].back();
		worklist_p--;
	}
	del_number--;
}

std::vector<int> QuadricSimplification::getAllTrianglesId(int u) {
	std::vector<int> ans;
	std::function<void(int)> dfs;
	dfs = [&](int u) {
		if(u < v_adj_tri.size())
			for (auto tri_id : v_adj_tri[u]) {
				ans.push_back(tri_id);
			}
		
		if (u < sons.size())
			for (auto v : sons[u]) {
				dfs(v);
			}
	};
	dfs(u);
	return ans;
}

void QuadricSimplification::init(std::shared_ptr<HEMeshX> heMesh) {
	int n = heMesh->Vertices().size();
	
	this->heMesh = heMesh;
	origin_vertics_number = n;
	del_number = 0;
	sons.clear();
	checkpoints.clear();
	worklist.clear();
	for (int i = 0; i < n; i++) {
		sons.push_back(std::vector<int>());
	}

	struct PairContraction {
		float cost;
		int i, j;
		pointf3 v_bar;
		bool operator <(const PairContraction& o) const {
			return cost > o.cost;
		}
	};

	std::vector<Eigen::Matrix4f> Qs;
	std::priority_queue<PairContraction> heap;

	v_adj_tri.clear();
	triangles.clear();
	for (auto* v : heMesh->Vertices()) {
		std::vector<int> tmp;
		for (auto* m : v->AdjPolygons()) 
			if(m != nullptr)
				tmp.push_back(heMesh->Index(m));
		v_adj_tri.push_back(tmp);
	}
	for (auto* m : heMesh->Polygons()) {
		Triangle tri;
		int t = 0;
		for (auto* v : m->AdjVertices()) tri.add_poschange_and_go(t++, heMesh->Index(v));
		triangles.push_back(tri);
	}

	// calc Q
	for (auto* U : heMesh->Vertices()) {
		auto Vs = U->AdjVertices();
		Eigen::Matrix4f Q = Eigen::Matrix4f::Zero();
		for (size_t i = 0; i < Vs.size(); i++) {
			Vertex* V = Vs[i];
			Vertex* W = Vs[(i + 1) % Vs.size()];
			float a, b, c, d;
			getPlane(U->position, V->position, W->position, a, b, c, d);
			Eigen::Vector4f p;
			p << a, b, c, d;
			Q += p * p.transpose();
		}
		Qs.push_back(Q);
	}
	spdlog::info("calc Q success");
	
	// push into heap
	for (auto* e : heMesh->Edges()) {
		auto* v1 = e->HalfEdge()->End();
		auto* v2 = e->HalfEdge()->Origin();
		int i = heMesh->Index(v1), j = heMesh->Index(v2);
		auto ans = getCostAndV(Qs[i] + Qs[j], v1->position, v2->position);
		heap.push(PairContraction{ ans.first, i, j, ans.second});
	}
	spdlog::info("push into heap success");

	// get opt sequence
	std::vector<bool> is_del(n, false);
	for (int o = 0; o < n - min_edge; o++) {
		PairContraction ele;
		int i, j;
		do {
			ele = heap.top(); heap.pop();
			i = ele.i, j = ele.j;
		}while (is_del[i] || is_del[j]);

		del_number++;

		int new_id = Qs.size();

		// record
		Qs.push_back(Qs[i] + Qs[j]);
		heMesh->AddVertex(); heMesh->Vertices().at(new_id)->position = ele.v_bar;
		sons.push_back({ i, j });
		is_del[i] = is_del[j] = true; is_del.push_back(false);
		checkpoints.push_back(worklist.size());

		std::set<std::pair<int, int>> edges;
		auto work_son = [&](int x) {
			std::vector<int> all_tri_id = getAllTrianglesId(x);
			for (int tri_id : all_tri_id) {
				if (!triangles[tri_id].is_valid()) continue;
				triangles[tri_id].add_vchange_and_go(x, new_id);
				worklist.push_back(tri_id);
				auto nei_ids = triangles[tri_id].get_nei(new_id);
				for (int i : nei_ids) {
					if (i == new_id) continue;
					edges.insert({ i, new_id });
				}
			}
		};

		work_son(i);
		work_son(j);
		
		// add new edges
		for (auto e : edges) {
			int i = e.first;
			int j = e.second;
			auto ans = getCostAndV(Qs[i] + Qs[j], heMesh->Vertices().at(i)->position, heMesh->Vertices().at(j)->position);
			heap.push(PairContraction{ ans.first, i, j, ans.second});
		}
	}
	checkpoints.push_back(worklist.size());

	worklist_p = (int)worklist.size() - 1;
	while (del_number) {
		contractInv();
	}
	
	is_init = true;
	spdlog::info("QEM init success");
}

bool QuadricSimplification::trans(float& rate) {
	int ori_n = origin_vertics_number;
	int n = origin_vertics_number * rate;
	n = std::min(n, ori_n);
	n = std::max(n, min_edge);

	int goal_del = ori_n - n;

	if (goal_del == del_number) return false;

	spdlog::info("del number: {} -> {}", del_number, goal_del);

	while (del_number < goal_del) {
		contract();
	}
	while (del_number > goal_del) {
		contractInv();
	}
	return true;
}

bool QuadricSimplification::hasInitialized() {
	return is_init;
}

std::vector<uint32_t> QuadricSimplification::getTriangles() {
	std::vector<uint32_t> ans;
	for (auto& t : triangles) {
		if (t.is_valid()) {
			for(int i = 0;i < 3;i ++)
				ans.push_back(t.id[i]);
		}
	}
	return ans;
}
