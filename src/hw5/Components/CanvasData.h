#pragma once

#include <UGM/UGM.h>
#include <python.h>

#include <spdlog/spdlog.h>

#include <_deps/imgui/imgui.h>

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Ubpa;


struct CanvasData {
	std::vector<Ubpa::pointf2> points_input;
	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };
	
	bool opt_enable_lines{ true };
	ImU32 color_lines{ IM_COL32(0, 255, 255, 255) };
	ImU32 color_curve{ IM_COL32(255, 0, 255, 255) };

	bool is_initialize{ false };

	int subdiv_indix{ 0 };
	bool is_curve_closed{ false };
	float alpha{ 0.1 };
	int subdiv_times{ 0 };

	bool initialize() {		
		spdlog::set_pattern("[%L] %v");
		
		spdlog::info("Initialize successfully");
		return true;
	}

	void push_point(Ubpa::pointf2 p) {
		points_input.push_back(p);
	}

	void clear_points() {
		points_input.clear();
	}

	void del_last_point() {
		points_input.resize(points_input.size() - 1);
	}
};



#include "details/CanvasData_AutoRefl.inl"