#pragma once

#include <UGM/UGM.h>

struct CanvasData {
	std::vector<Ubpa::pointf2> points;
	std::vector<Ubpa::pointf2> points_poly;
	std::vector<Ubpa::pointf2> points_Gauss;
	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };
	
	bool opt_enable_lines{ true };
	bool opt_enable_polynomial{ true };
	
	bool opt_enable_Gauss{ true };
	float sigma_Gauss{ 0.1f };

	bool opt_enable_least_squares{ true };
	int n_ls{ 1 };

	bool opt_enable_ridge{ true };
	int n_ridge{ 1 };
	float lambda_ridge{ 0.1f };

	bool adding_line{ false };

	void pushPoint(Ubpa::pointf2 p) {
		int index = points.size();
		for (int i = 0; i < points.size(); i++) {
			if (points[i][0] > p[0]) {
				index = i;
				break;
			}
		}
		points.insert(points.begin() + index, p);
	}
};

#include "details/CanvasData_AutoRefl.inl"