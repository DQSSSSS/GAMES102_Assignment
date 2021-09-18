#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <Python.h>

#include <spdlog/spdlog.h>

using namespace Ubpa;

#define DEBUG(x) ( !(x) && (spdlog::error(("error at line "+ std::to_string(__LINE__))), 0) )

void draw(ImDrawList* draw_list, const std::vector<pointf2>& points0, ImVec2 origin, ImU32 color, bool is_close) {
	auto points = points0;
	if (is_close && points.size() >= 1) {
		points.push_back(points[0]);
	}
	for (size_t n = 0; n + 1 < points.size(); n++)
		draw_list->AddLine(ImVec2(origin.x + points[n][0], origin.y + points[n][1]), 
			ImVec2(origin.x + points[n + 1][0], origin.y + points[n + 1][1]), color, 2.0f);
}

pointf2 add(pointf2 a, pointf2 b) { return pointf2(a[0] + b[0], a[1] + b[1]); }
pointf2 mul(float t, pointf2 a) { return pointf2(t * a[0], t * a[1]); }

std::vector<pointf2> get_Chaikin(const std::vector<pointf2>& points, int n, bool is_closed) {
	if (points.size() <= 1) return {};
	std::vector<pointf2> ans = points;
	while(n --) {
		std::vector<pointf2> tmp;
		auto work = [&](pointf2 a, pointf2 b) {
			auto ret_1 = add(mul(3.0 / 4, a), mul(1.0 / 4, b));
			auto ret_2 = add(mul(1.0 / 4, a), mul(3.0 / 4, b));
			tmp.push_back(ret_1);
			tmp.push_back(ret_2);
		};
		auto pr = [&](int id) { return id == 0 ? ans.size() - 1 : id - 1; };
		auto nx = [&](int id) { return id + 1 == ans.size() ? 0 : id + 1; };
		for (size_t i = 0; i < ans.size(); i++) {
			int t0 = i, t1 = nx(i);
			if(is_closed || (t0 < t1))
				work(ans[t0], ans[t1]);
		}
		//if (is_closed) {
		//	int t = ans.size();
		//	work(ans[t - 1], ans[0]);
		//}
		ans = tmp;
	}
	return ans;
}

std::vector<pointf2> get_Cubic(const std::vector<pointf2>& points, int n, bool is_closed) {
	if (points.size() <= 1) return {};
	auto ans = points;
	while (n--) {
		std::vector<pointf2> tmp;
		auto work3 = [&](pointf2 a, pointf2 b, pointf2 c) {
			auto ret_1 = add(add(mul(1.0 / 8, a), mul(3.0 / 4, b)), mul(1.0 / 8, c));
			tmp.push_back(ret_1);
		};
		auto work2 = [&](pointf2 a, pointf2 b) {
			auto ret_1 = add(mul(1.0 / 2, a), mul(1.0 / 2, b));
			tmp.push_back(ret_1);
		};
		auto pr = [&](int id) { return id == 0 ? ans.size() - 1 : id - 1; };
		auto nx = [&](int id) { return id + 1 == ans.size() ? 0 : id + 1; };
		for (size_t i = 0; i < ans.size(); i++) {
			int t0, t1, t2;
			t0 = pr(i), t1 = i, t2 = nx(i);
			if (is_closed || (t0 < t1 && t1 < t2))
				work3(ans[t0], ans[t1], ans[t2]);
			t0 = i, t1 = nx(i);
			if(is_closed || (t0 < t1))
				work2(ans[t0], ans[t1]);
		}
		//if (is_closed) {

		//	int t = ans.size() - 1;
		//	work3(ans[pr(t)], ans[t], ans[nx(t)]);
		//	work2(ans[t], ans[nx(t)]);
		//	work3(ans[t], ans[nx(t)], ans[nx(nx(t))]);
		//}
		ans = tmp;
	}
	return ans;
}

std::vector<pointf2> get_4point(const std::vector<pointf2>& points, int n, float alpha, bool is_closed) {
	if (points.size() <= 1) return {};
	auto ans = points;
	while (n--) {
		std::vector<pointf2> tmp;
		auto work = [&](pointf2 a, pointf2 b, pointf2 c, pointf2 d) {
			auto A = add(mul(1.0 / 2, b), mul(1.0 / 2, c));
			auto B = add(mul(-1.0 / 2, a), mul(-1.0 / 2, d));
			auto ret = add(A, mul(alpha, add(A, B)));
			tmp.push_back(ret);
		};
		auto pr = [&](int id) { return id == 0 ? ans.size() - 1 : id - 1; };
		auto nx = [&](int id) { return id + 1 == ans.size() ? 0 : id + 1; };
		for (size_t i = 0; i < ans.size(); i++) {
			tmp.push_back(ans[i]);
			int t0 = pr(i), t1 = i, t2 = nx(i), t3 = nx(nx(i));
			if (is_closed || (t0 < t1 && t1 < t2 && t2 < t3))
				work(ans[t0], ans[t1], ans[t2], ans[t3]);
		}
		ans = tmp;
	}
	return ans;
}

void CanvasSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<CanvasData>();
		if (!data)
			return;

		if (ImGui::Begin("Canvas")) {

			if (!data->is_initialize) {
				data->is_initialize = data->initialize();
			}

			//spdlog::info("haha"); https://github.com/gabime/spdlog

			ImGuiStyle& style = ImGui::GetStyle();
			// below is just a random color, as example
			ImGui::Checkbox("Enable grid", &data->opt_enable_grid);
			ImGui::SameLine();
			ImGui::Checkbox("Enable context menu", &data->opt_enable_context_menu);

			ImGui::PushStyleColor(ImGuiCol_Text, data->color_lines);
			ImGui::Checkbox("Draw lines", &data->opt_enable_lines); // line
			ImGui::PopStyleColor();

			ImGui::RadioButton("Chaikin", &data->subdiv_indix, 0); // 
			ImGui::SameLine();
			ImGui::RadioButton("Cubic", &data->subdiv_indix, 1); // 
			ImGui::SameLine();
			ImGui::RadioButton("4-point interpolatory", &data->subdiv_indix, 2); // 
			
			ImGui::Checkbox("Closed", &data->is_curve_closed); 

			ImGui::SliderInt("Subdiv times", &data->subdiv_times, 0, 20); // uniform
			
			if (data->subdiv_indix == 2) {
				ImGui::SliderFloat("Alpha", &data->alpha, 0.0f, 1.0f); // uniform
			}

			std::string debug_info;


			ImGui::Text((
			debug_info	
			).c_str());

			//std::string test_hint = test(); ImGui::Text(test_hint.c_str());

			ImGui::Text("Mouse Left: drag to add lines,\nMouse Right: drag to scroll, click for context menu.");

			// Typically you would use a BeginChild()/EndChild() pair to benefit from a clipping region + own scrolling.
			// Here we demonstrate that this can be replaced by simple offsetting + custom drawing + PushClipRect/PopClipRect() calls.
			// To use a child window instead we could use, e.g:
			//      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));      // Disable padding
			//      ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(50, 50, 50, 255));  // Set a background color
			//      ImGui::BeginChild("canvas", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_NoMove);
			//      ImGui::PopStyleColor();
			//      ImGui::PopStyleVar();
			//      [...]
			//      ImGui::EndChild();

			// Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2) allows us to use IsItemHovered()/IsItemActive()
			ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      // ImDrawList API uses screen coordinates!
			ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   // Resize canvas to what's available
			if (canvas_sz.x < 50.0f) canvas_sz.x = 50.0f;
			if (canvas_sz.y < 50.0f) canvas_sz.y = 50.0f;
			ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

			// Draw border and background color
			ImGuiIO& io = ImGui::GetIO();
			ImDrawList* draw_list = ImGui::GetWindowDrawList();
			draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
			draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

			// This will catch our interactions
			ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
			const bool is_hovered = ImGui::IsItemHovered(); // Hovered
			const bool is_active = ImGui::IsItemActive();   // Held
			const ImVec2 origin(canvas_p0.x + data->scrolling[0], canvas_p0.y + data->scrolling[1]); // Lock scrolled origin
			const pointf2 mouse_pos_in_canvas(io.MousePos.x - origin.x, io.MousePos.y - origin.y);


			// Context menu (under default mouse threshold)
			ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
			if (data->opt_enable_context_menu && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && drag_delta.x == 0.0f && drag_delta.y == 0.0f)
				ImGui::OpenPopupContextItem("context");
			if (ImGui::BeginPopup("context")) {
				//if (ImGui::MenuItem("Remove one", NULL, false, data->points.size() > 0)) { data->points.resize(data->points.size() - 1); }
				if (ImGui::MenuItem("Remove all", NULL, false, data->points_input.size() > 0)) { data->clear_points(); }
				if (ImGui::MenuItem("Remove one", NULL, false, data->points_input.size() > 0)) { data->del_last_point(); }
				ImGui::EndPopup();
			}

			// Pan (we use a zero mouse threshold when there's no context menu)
				// You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
			const float mouse_threshold_for_pan = data->opt_enable_context_menu ? -1.0f : 0.0f;
			if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan)) {
				data->scrolling[0] += io.MouseDelta.x;
				data->scrolling[1] += io.MouseDelta.y;
			}
			if (is_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
				data->push_point(mouse_pos_in_canvas);
			}

			// Draw grid + all lines in the canvas
			draw_list->PushClipRect(canvas_p0, canvas_p1, true);

			if (data->opt_enable_grid)
			{
				const float GRID_STEP = 64.0f;
				for (float x = fmodf(data->scrolling[0], GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 40));
				for (float y = fmodf(data->scrolling[1], GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 40));
			}

			
			// ---------------------

			// Draw lines
			if (data->opt_enable_lines) {
				draw(draw_list, data->points_input, origin, data->color_lines, data->is_curve_closed);
			}

			std::vector<pointf2> curve;
			if (data->subdiv_indix == 0) { // 2 point
				curve = get_Chaikin(data->points_input, data->subdiv_times, data->is_curve_closed);
			}
			else if (data->subdiv_indix == 1) {// 3
				curve = get_Cubic(data->points_input, data->subdiv_times, data->is_curve_closed);
			}
			else if (data->subdiv_indix == 2) {
				curve = get_4point(data->points_input, data->subdiv_times, data->alpha, data->is_curve_closed);
			}

			draw(draw_list, curve, origin, data->color_curve, data->is_curve_closed);

			// ---------------------


			// Draw points
			const float point_radius = 5.0f;
			for (int n = 0; n < data->points_input.size(); n++) {
				draw_list->AddCircleFilled(ImVec2(data->points_input[n][0] + origin.x, data->points_input[n][1] + origin.y), point_radius, IM_COL32(255, 255, 255, 255));
			}
			draw_list->AddCircleFilled(ImVec2(mouse_pos_in_canvas[0] + origin.x, mouse_pos_in_canvas[1] + origin.y), point_radius, IM_COL32(255, 0, 0, 255));

			draw_list->PopClipRect();
		}

		ImGui::End();
		});
}