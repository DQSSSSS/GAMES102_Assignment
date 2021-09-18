#pragma once

#include "../HEMeshX.h"

#include <queue>

using namespace Ubpa;

class QuadricSimplification {

public:

	void init(std::shared_ptr<HEMeshX> heMesh);

	bool trans(float& rate);

	bool hasInitialized();

	std::vector<uint32_t> getTriangles();

private:

	void contract();
	void contractInv();
	std::vector<int> getAllTrianglesId(int u);
	
	std::shared_ptr<HEMeshX> heMesh = nullptr;
	int origin_vertics_number;
	bool is_init = false;

	struct Triangle {
		int id[3] = {-1, -1, -1};
		int nowp = 0;
		std::vector<std::pair<int, std::pair<int, int>>> change_list;

		void add_poschange_and_go(int pos, int x) { 
			change_list.push_back({ pos, {id[pos], x} });
			go();
		}

		void add_vchange_and_go(int v, int x) {
			add_poschange_and_go(get_id(v), x);
		}

		void go() { 	
			id[change_list[nowp].first] = change_list[nowp].second.second;
			nowp++;
		}

		void back() {
			assert(nowp > 3);
			nowp--;
			id[change_list[nowp].first] = change_list[nowp].second.first;
		}

		bool is_valid() { return id[0] != id[1] && id[0] != id[2] && id[1] != id[2]; }

		std::vector<int> get_nei(int x) {
			if (x == id[0]) return { id[1], id[2] };
			if (x == id[1]) return { id[0], id[2] };
			if (x == id[2]) return { id[0], id[1] };
			assert(0);
		}

		int get_id(int x) {
			if (x == id[0]) return 0;
			if (x == id[1]) return 1;
			if (x == id[2]) return 2;
			assert(0);
		}
	};

	std::vector<Triangle> triangles;
	std::vector<std::vector<int>> v_adj_tri;
	std::vector<int> checkpoints;
	std::vector<int> worklist;
	std::vector<std::vector<int>> sons;
	int del_number, worklist_p;

	static const int min_edge = 5;
};