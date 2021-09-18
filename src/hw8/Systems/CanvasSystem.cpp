#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>

#include <spdlog/spdlog.h>

#define JCV_REAL_TYPE double
#define JC_VORONOI_IMPLEMENTATION
#include <voronoi/src/jc_voronoi.h>

using namespace Ubpa;

float random() {
	std::random_device rd;  // 将用于获得随机数引擎的种子
	std::mt19937 gen(rd()); // 以 rd() 播种的标准 mersenne_twister_engine
	std::uniform_real_distribution<> dis(0, 1);
	return dis(gen);
}

float random_lr(int l, int r) {
	return l + (r - l) * random();
}


class CVT {

public:

	const double INF = 1e9;

	jcv_diagram            diagram_{ 0 };
	std::vector<jcv_point> points_;
	jcv_rect               bbox_;
	double				   diff_lst{ INF };

	void init(const std::vector<pointf2>& points, valf2 position, valf2 size) {
		points_.clear();
		points_.resize(points.size());
		for (size_t i = 0; i < points_.size(); i++) {
			points_[i].x = points[i][0];
			points_[i].y = points[i][1];
		}
		bbox_.min.x = position[0];
		bbox_.min.y = position[1];
		bbox_.max.x = position[0] + size[0];
		bbox_.max.y = position[1] + size[1];
		diff_lst = 1e9;
	}

	~CVT() { if (diagram_.internal) jcv_diagram_free(&diagram_); }

	jcv_point polygon_centroid(const jcv_site* site) {
		auto method1 = [&]() {
			double total_det = 0;
			jcv_point center{ 0, 0 };
			for (auto e = site->edges; e; e = e->next) {
				jcv_point p1 = e->pos[0], p2 = e->pos[1];
				double det = p1.x * p2.y - p2.x * p1.y;
				total_det += det;
				center.x += (p1.x + p2.x) * det;
				center.y += (p1.y + p2.y) * det;
			}
			center.x /= 3 * total_det;
			center.y /= 3 * total_det;
			return center;
		};
		auto method2 = [&]() {
			int cnt = 1;
			auto center = site->p;
			for (auto e = site->edges; e; e = e->next) {
				center.x += e->pos[0].x;
				center.y += e->pos[0].y;
				cnt++;
			}
			center.x = center.x / cnt;
			center.y = center.y / cnt;
			return center;
		};
		return method2();
	}

	double relax_points() {
		const jcv_site* sites = jcv_diagram_get_sites(&diagram_);
		double max_diff = 1e-9;
		//spdlog::info("diagram numsites: {}", diagram_.numsites);
		for (int i = 0; i < diagram_.numsites; ++i) {
			const jcv_site* site = &sites[i];
			const jcv_point pre_p = points_[site->index];
			points_[site->index] = polygon_centroid(site);
			//spdlog::info("source: ({}, {}), target: ({}, {})", pre_p.x, pre_p.y, points_[site->index].x, points_[site->index].y);
			max_diff = std::max(max_diff, jcv_point_dist_sq(&pre_p, &points_[site->index]));
		}
		return max_diff;
	}

	double Lloyd(bool is_relax = true) {
		if (diff_lst < 1e-4) return diff_lst;
		if (diagram_.internal) jcv_diagram_free(&diagram_);
		memset(&diagram_, 0, sizeof(jcv_diagram));
		jcv_diagram_generate(points_.size(), points_.data(), &bbox_, 0, &diagram_);
		if (!is_relax) return diff_lst;
		double diff = relax_points();
		spdlog::info("Lloyd success, diff = {}", diff);
		diff_lst = diff;
		return diff;
	}

	std::vector<pointf2> get_points() {
		std::vector<pointf2> points;
		points.resize(points_.size());
		for (size_t i = 0; i < points_.size(); i++) {
			points[i][0] = points_[i].x;
			points[i][1] = points_[i].y;
		}
		return points;
	}

	std::vector<std::pair<pointf2, pointf2>> get_Voronoi() {
		std::vector<std::pair<pointf2, pointf2>> ans;
		if (diff_lst >= INF) return ans;
		Lloyd(false);
		auto add_edge = [&](jcv_point a, jcv_point b) {
			ans.push_back({ pointf2(a.x, a.y), pointf2(b.x, b.y) });
		};
		const jcv_edge* edge = jcv_diagram_get_edges(&diagram_);
		while (edge) {
			add_edge(edge->pos[0], edge->pos[1]);
			edge = jcv_diagram_get_next_edge(edge);
		}
		return ans;
	}

	std::vector<std::pair<pointf2, pointf2>> get_triangle() {
		std::vector<std::pair<pointf2, pointf2>> ans;
		if (diff_lst >= INF) return ans;
		Lloyd(false);
		auto add_edge = [&](jcv_point a, jcv_point b) {
			ans.push_back({ pointf2(a.x, a.y), pointf2(b.x, b.y) });
		};
		
		std::map<std::pair<pointf2, pointf2>, std::vector<pointf2>> mp;

		const jcv_site* sites = jcv_diagram_get_sites(&diagram_);
		for (int i = 0; i < diagram_.numsites; ++i) {
			const jcv_site* site = &sites[i];
			const jcv_graphedge* e = site->edges;
			while (e) {
				auto p0 = pointf2(e->pos[0].x, e->pos[0].y);
				auto p1 = pointf2(e->pos[1].x, e->pos[1].y);
				if (p0[0] > p1[0]) swap(p0, p1);
				//mp[{p0, p1}].push_back(pointf2(points_[i].x, points_[i].y));
				mp[{p0, p1}].push_back(pointf2(site->p.x, site->p.y));
				//add_edge(site->p, e->pos[0]);
				//add_edge(e->pos[0], e->pos[1]);
				//add_edge(e->pos[1], site->p);
				//draw_triangle(site->p, e->pos[0], e->pos[1]);
				e = e->next;
			}
		}
		for (auto& p : mp) {
			if (p.second.size() == 2) {
				ans.push_back({ p.second[0], p.second[1] });
				//spdlog::info("border: ({}, {}),  ({}, {})", p.first.first[0], p.first.first[1], p.first.second[0], p.first.second[1]);
				//spdlog::info("point: ({}, {}),  ({}, {})", p.second[0][0], p.second[0][1], p.second[1][0], p.second[1][1]);
			}
		}
		//spdlog::info("Trianglation success");
		return ans;
	}
}cvt;



void CanvasSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<CanvasData>();
		if (!data)
			return;

		if (ImGui::Begin("Canvas")) {
			ImGui::Checkbox("Enable grid", &data->opt_enable_grid);
			//ImGui::Checkbox("Enable context menu", &data->opt_enable_context_menu);

			ImGui::SetNextItemWidth(100);
			ImGui::InputInt("Points number", &data->points_number);
			ImGui::SameLine();
			if (ImGui::Button("Random Generate Points")) {
				data->points.clear();
				for (int i = 0; i < data->points_number; i++) {
					float x = data->panel_pos[0] + random_lr(0, data->panel_size[0]);
					float y = data->panel_pos[1] + random_lr(0, data->panel_size[1]);
					data->points.push_back(pointf2(x, y));
				}
				cvt.init(data->points, data->panel_pos, data->panel_size);
			}

			ImGui::Checkbox("Voronoi Grid", &data->is_draw_Voronoi);
			ImGui::SameLine();
			ImGui::Checkbox("Delaunay triangulation", &data->is_draw_triangle);
			
			if (ImGui::Button("Lloyd Method")) {
				for (int i = 0; i < data->Lloyd_times; i++) {
					cvt.Lloyd();
					data->points = cvt.get_points();
				}
			}
			ImGui::SameLine();
			ImGui::SetNextItemWidth(100);
			ImGui::InputInt("Iteration times", &data->Lloyd_times);
			ImGui::SameLine();
			if (ImGui::Button(data->is_Lloyd_running ? "Stop" : "Auto")) {
				data->is_Lloyd_running ^= 1;
			}

			if (data->is_Lloyd_running) {
				cvt.Lloyd();
				data->points = cvt.get_points();
			}

			ImGui::Text((
				"(" + std::to_string(data->mouse_point[0]) + ", " + std::to_string(data->mouse_point[1]) + ")"
				).c_str());

//			ImGui::Text("Mouse Left: drag to add lines,\nMouse Right: drag to scroll, click for context menu.");



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

			// Add first and second point

			data->mouse_point = mouse_pos_in_canvas;

			// Pan (we use a zero mouse threshold when there's no context menu)
			// You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
			/*const float mouse_threshold_for_pan = data->opt_enable_context_menu ? -1.0f : 0.0f;
			if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan))
			{
				data->scrolling[0] += io.MouseDelta.x;
				data->scrolling[1] += io.MouseDelta.y;
			}*/

			// Context menu (under default mouse threshold)
			/*ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
			if (data->opt_enable_context_menu && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && drag_delta.x == 0.0f && drag_delta.y == 0.0f)
				ImGui::OpenPopupContextItem("context");
			if (ImGui::BeginPopup("context"))
			{
				if (data->adding_line)
					data->points.resize(data->points.size() - 2);
				data->adding_line = false;
				if (ImGui::MenuItem("Remove one", NULL, false, data->points.size() > 0)) { data->points.resize(data->points.size() - 2); }
				if (ImGui::MenuItem("Remove all", NULL, false, data->points.size() > 0)) { data->points.clear(); }
				ImGui::EndPopup();
			}*/

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

			draw_list->AddRectFilled(data->panel_pos + origin, data->panel_pos + data->panel_size + origin, IM_COL32(255, 255, 255, 200));

			const float point_radius = 3.0f;
			for (int n = 0; n < data->points.size(); n++)
				draw_list->AddCircleFilled(ImVec2(data->points[n][0] + origin.x, data->points[n][1] + origin.y), 
					point_radius, IM_COL32(0, 0, 0, 255));

			if (data->is_draw_Voronoi) {
				auto edges = cvt.get_Voronoi();
				for (auto p : edges) {
					auto p1 = ImVec2(p.first[0] + origin.x, p.first[1] + origin.y);
					auto p2 = ImVec2(p.second[0] + origin.x, p.second[1] + origin.y);
					draw_list->AddLine(p1, p2, IM_COL32(0, 0, 255, 255));
				}
			}

			if (data->is_draw_triangle) {
				auto edges = cvt.get_triangle();
				for (auto p : edges) {
					auto p1 = ImVec2(p.first[0] + origin.x, p.first[1] + origin.y);
					auto p2 = ImVec2(p.second[0] + origin.x, p.second[1] + origin.y);
					draw_list->AddLine(p1, p2, IM_COL32(255, 0, 0, 255));
				}
			}
			
			draw_list->PopClipRect();
		}

		ImGui::End();
	});
}
