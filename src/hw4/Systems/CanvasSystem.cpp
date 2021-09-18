#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <Python.h>

#include <spdlog/spdlog.h>

using namespace Ubpa;

#define DEBUG(x) ( !(x) && (spdlog::error(("error at line "+ std::to_string(__LINE__))), 0) )

void draw(ImDrawList* draw_list, const std::vector<pointf2>& points, ImVec2 origin, ImU32 color) {
	for (size_t n = 0; n + 1 < points.size(); n++)
		draw_list->AddLine(ImVec2(origin.x + points[n][0], origin.y + points[n][1]), 
			ImVec2(origin.x + points[n + 1][0], origin.y + points[n + 1][1]), color, 2.0f);
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

			ImGui::RadioButton("Spline", &data->curve_indix, 0); // spline
			ImGui::SameLine();
			ImGui::RadioButton("Bezier", &data->curve_indix, 1); // Bezier
			
			ImGui::Checkbox("Edit curve", &data->is_editing); // params

			ImGui::RadioButton("Uniform", &data->params_indix, 0); // uniform
			ImGui::SameLine();
			ImGui::RadioButton("Chordal", &data->params_indix, 1); // chordal
			ImGui::SameLine();
			ImGui::RadioButton("Centripetal", &data->params_indix, 2); // centripetal
			ImGui::SameLine();
			ImGui::RadioButton("Foley", &data->params_indix, 3); // Foley



			std::string debug_info;
			for (auto p : data->get_all_input_points()) {
				debug_info += my_tools::get_p_info(p);
			}
			debug_info += "\n";
			debug_info += std::to_string(data->get_focus_point_id());
			debug_info += " " + std::to_string(data->get_focus_control_point_j());

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

			if (!data->is_editing) {

				if (!data->build_curve()) {
					spdlog::error("Build curve error");
				}

				if (is_hovered) {
					data->point_now = mouse_pos_in_canvas;
				}

				if (is_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
					data->push_point(mouse_pos_in_canvas);
					data->is_drawing_point = true;
				}

				if (is_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
					spdlog::info("End curve");
					data->is_drawing_point = false;
				}

				// Context menu (under default mouse threshold)
				ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
				if (data->opt_enable_context_menu && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && drag_delta.x == 0.0f && drag_delta.y == 0.0f)
					ImGui::OpenPopupContextItem("context");
				if (ImGui::BeginPopup("context")) {
					//if (ImGui::MenuItem("Remove one", NULL, false, data->points.size() > 0)) { data->points.resize(data->points.size() - 1); }
					if (ImGui::MenuItem("Remove all", NULL, false, !data->is_editing && data->points_input.size() > 0)) { data->clear_points(); }
					if (ImGui::MenuItem("Remove one", NULL, false, !data->is_editing && data->points_input.size() > 0)) { data->del_last_point(); }
					ImGui::EndPopup();
				}
			
			}
			else {
				data->is_drawing_point = false;

				if (is_hovered && (ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::IsMouseClicked(ImGuiMouseButton_Right))) {
					data->found_control_point(mouse_pos_in_canvas);
					//spdlog::info(get_p_info(mouse_pos_in_canvas));
				}

				if (is_hovered && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0)) {
					//spdlog::info("left dragging");
					data->move_control_point(io.MouseDelta.x, io.MouseDelta.y);
				}

				ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
				if (ImGui::IsMouseReleased(ImGuiMouseButton_Right) && drag_delta.x == 0.0f && drag_delta.y == 0.0f)
					ImGui::OpenPopupContextItem("context");
				if (ImGui::BeginPopup("context")) {
					if (ImGui::MenuItem("Smooth vertex(C1)", NULL, false, data->get_focus_point_id() != -1)) {
						spdlog::info("C1");
						data->set_point_type("S");
					}	
					if (ImGui::MenuItem("Right vertex(G1)", NULL, false, data->get_focus_point_id() != -1)) {
						spdlog::info("G1");
						data->set_point_type("R");
					}
					if (ImGui::MenuItem("Corner vertex(C0/G0)", NULL, false, data->get_focus_point_id() != -1)) {
						spdlog::info("C0");
						data->set_point_type("C");
					}
					ImGui::EndPopup();
				}
			}

			// Pan (we use a zero mouse threshold when there's no context menu)
				// You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
			const float mouse_threshold_for_pan = data->opt_enable_context_menu ? -1.0f : 0.0f;
			if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan)) {
				data->scrolling[0] += io.MouseDelta.x;
				data->scrolling[1] += io.MouseDelta.y;
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

			//const float lb = data->points[0][0] - 5;
			//const float rb = data->points[data->points.size()-1][0] + 5;
			const float lb = 0;
			const float rb = 1;
			const int number = 1000;

			// Draw lines
			if (data->opt_enable_lines) {
				draw(draw_list, data->get_all_input_points(), origin, data->color_lines);
			}

			draw(draw_list, data->get_curve(lb, rb, number), origin, data->color_lines);

			// ---------------------


			// Draw points
			const float point_radius = 5.0f;
			for (int n = 0; n < data->points_input.size(); n++) {
				if (!data->is_editing) {
					draw_list->AddCircleFilled(ImVec2(data->points_input[n][0] + origin.x, data->points_input[n][1] + origin.y), point_radius, IM_COL32(255, 255, 255, 255));
				}
				else {

					auto get_shift_xy = [](pointf2 p, float dx, float dy) -> ImVec2 {
						return ImVec2(p[0] + dx, p[1] + dy);
					};

//					auto p_min = ImVec2(data->points_input[n][0] + origin.x - point_radius, data->points_input[n][1] + origin.y - point_radius);
//					auto p_max = ImVec2(data->points_input[n][0] + origin.x + point_radius, data->points_input[n][1] + origin.y + point_radius);
					auto p_min = get_shift_xy(data->points_input[n], origin.x - point_radius, origin.y - point_radius);
					auto p_max = get_shift_xy(data->points_input[n], origin.x + point_radius, origin.y + point_radius);
					draw_list->AddRectFilled(p_min, p_max, IM_COL32(0, 0, 0, 255));
					draw_list->AddRect(p_min, p_max, IM_COL32(255, 255, 255, 255));

					if (n == data->get_focus_point_id()) {
						
						auto draw_cp = [&](pointf2 o, pointf2 cp) {
							draw_list->AddLine(get_shift_xy(o, origin.x, origin.y),
								get_shift_xy(cp, origin.x, origin.y), IM_COL32(0, 0, 0, 255), 1.0f);
							auto p_min = get_shift_xy(cp, origin.x - point_radius, origin.y - point_radius);
							auto p_max = get_shift_xy(cp, origin.x + point_radius, origin.y + point_radius);
							draw_list->AddRectFilled(p_min, p_max, IM_COL32(0, 0, 0, 255));
							draw_list->AddRect(p_min, p_max, IM_COL32(255, 255, 0, 255));
						};

						auto cps = data->get_control_points(n);
						//spdlog::info("f: " + my_tools::get_p_info(data->points_input[n]));
						for (size_t i = 0; i < cps.size(); i++) {
						//	spdlog::info("cps: " + my_tools::get_p_info(cps[i]));
							draw_cp(data->points_input[n], cps[i]);
						}
					}

				}
			}
			draw_list->AddCircleFilled(ImVec2(mouse_pos_in_canvas[0] + origin.x, mouse_pos_in_canvas[1] + origin.y), point_radius, IM_COL32(255, 0, 0, 255));

			draw_list->PopClipRect();
		}

		ImGui::End();
		});
}