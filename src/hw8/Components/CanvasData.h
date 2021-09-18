#pragma once

#include <UGM/UGM.h>

struct CanvasData {
	std::vector<Ubpa::pointf2> points;
	int points_number{ 10 };
	Ubpa::valf2 panel_pos{ 50, 30 };
	Ubpa::valf2 panel_size{ 500, 300 };
	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };

	bool is_Lloyd_running{ false };
	int Lloyd_times{ 1 };
	bool is_draw_triangle{ false };
	bool is_draw_Voronoi{ false };

	Ubpa::pointf2 mouse_point{ 0, 0 };

};

#include "details/CanvasData_AutoRefl.inl"
