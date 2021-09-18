#pragma once

#include <UGM/UGM.h>
#include <python.h>

#include <spdlog/spdlog.h>

#include <_deps/imgui/imgui.h>

#include "SplineCurve.h"
#include "tools.h"

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Ubpa;


struct CanvasData {
	CubicSplineCurve csc;
	std::vector<Ubpa::pointf2> points_input;

	Ubpa::pointf2 point_now; // mouse

	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };
	
	bool opt_enable_lines{ true };
	ImU32 color_lines{ IM_COL32(0, 255, 255, 255) };

	bool is_initialize{ false };

	bool is_drawing_point{ false };
	bool is_editing{ false };

	bool opt_enable_params{ false };
	int params_indix{ 0 };
	
	int curve_indix{ 0 }; // 0: spline, 1: Bezier

	bool initialize() {		
		spdlog::set_pattern("[%L] %v");
		
		spdlog::info("Initialize successfully");
		return true;
	}

	bool build_curve() {
		auto points = get_all_input_points();
		if (points.size() >= 2) {
			auto t_vec = my_tools::parameterize(points, params_indix);
			if (curve_indix == 0) { // spline
				if (!csc.build_curve(points, t_vec)) {
					spdlog::error("csc loaded error");
					return false;
				}
				return true;
			}
			else if (curve_indix == 1) { // Bezier

			}
		}
		return true;
	}

	bool found_control_point(Ubpa::pointf2 o) {
		if (curve_indix == 0) { // spline
			return csc.found_control_point(o);
		}
		else if (curve_indix == 1) { // Bezier

		}
		return false;
	}

	bool move_control_point(float dx, float dy) {
		if (curve_indix == 0) { // spline
			if (csc.move_control_point(dx, dy)) {
				points_input = csc.points;
				return true;
			}
			spdlog::error("move cp error");
			return false;
		}
		else if (curve_indix == 1) { // Bezier

		}
		return true;
	}

	std::vector<pointf2> get_curve(float lb = 0, float rb = 1, int number = 1000) {
		if (curve_indix == 0) { // spline
			return csc.get_curve(lb, rb, number);
		}
		else if (curve_indix == 1) { // Bezier

		}
		return {};
	}

	int get_focus_point_id() {
		if (curve_indix == 0) { // spline
			return csc.get_focus_point_id();
		}
		else if (curve_indix == 1) { // Bezier

		}
	}

	int get_focus_control_point_j() {
		if (curve_indix == 0) { // spline
			return csc.focus_cp_j;
		}
		else if (curve_indix == 1) { // Bezier

		}
	}

	std::vector<pointf2> get_control_points(int n) {
		if (curve_indix == 0) { // spline
			return csc.get_control_points(n);
		}
		else if (curve_indix == 1) { // Bezier

		}
		return {};
	}

	void set_point_type(std::string s) {
		if (curve_indix == 0) { // spline
			return csc.set_point_type(s);
		}
		else if (curve_indix == 1) { // Bezier

		}
	}

	void push_point(Ubpa::pointf2 p) {
		points_input.push_back(p);
	}

	std::vector<Ubpa::pointf2> get_all_input_points() {
		if (!is_drawing_point) return points_input;
		auto ans = points_input;	
		ans.push_back(point_now);
		return ans;
	}

	void clear_points() {
		points_input.clear();
	}

	void del_last_point() {
		points_input.resize(points_input.size() - 1);
	}
};



#include "details/CanvasData_AutoRefl.inl"