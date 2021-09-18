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

std::vector<pointf2> get_poly_results(const std::vector<pointf2>& points, float lb, float rb, int number) {
	if (points.size() <= 1) return {};

	auto f = [&](float x) {
		int n = points.size(); // a0 + a_1*x + ... + a_{n-1}*x^{n-1}
		float ans = 0;
		for (int i = 0; i < n; i++) {
			float tmp_fz = 1, tmp_fm = 1;
			for (int j = 0; j < n; j++) {
				if (i == j) continue;
				tmp_fz *= points[j][0] - x;
				tmp_fm *= points[j][0] - points[i][0];
			}
			ans += tmp_fz / tmp_fm * points[i][1];
		}
		return ans;
	};

	std::vector<pointf2> ans;
	for (int i = 0; i < number; i++) {
		float x = lb + (rb - lb) / number * i;
		ans.push_back(pointf2(x, f(x)));
	}
	return ans;
}

std::vector<pointf2> get_gauss_results(const std::vector<pointf2>& points, float sigma, float lb, float rb, int number) {
	if (points.size() <= 1) return {};

	auto g = [&](int i, float x) {
		if (i == 0) return 1.0f;
		float xi = points[i - 1][0];
		return std::exp(-(x - xi) * (x - xi) / (2 * sigma * sigma));
	};

	int n = points.size();

	float nx = 0, ny = 0; // new point
	for (int i = 0; i < n; i++) nx += points[i][0], ny += points[i][1];
	nx /= n, ny /= n;

	// Y=GA
	Eigen::MatrixXf G(n + 1, n + 1);
	for (int i = 0; i <= n; i++) {
		if (i < n) { // row in [0, n-1], use origin points
			for (int j = 0; j <= n; j++) {
				G(i, j) = g(j, points[i][0]);
			}
		}
		else {	// row=n
			for (int j = 0; j <= n; j++) {
				G(i, j) = g(j, nx);
			}
		}
	}

	Eigen::VectorXf Y(n + 1);
	for (int i = 0; i <= n; i++) {
		Y(i) = i < n ? points[i][1] : ny;
	}
	Eigen::VectorXf A = G.colPivHouseholderQr().solve(Y);

	// f(x) = a0 + a_1*g_1(x) + ... + a_n*g_n(x)
	auto f = [&](float x) {
		int n = points.size();
		float ans = 0;
		for (int i = 0; i <= n; i++) {
			ans += A[i] * g(i, x);
		}
		return ans;
	};

	std::vector<pointf2> ans;
	for (int i = 0; i < number; i++) {
		float x = lb + (rb - lb) / number * i;
		ans.push_back(pointf2(x, f(x)));
	}
	return ans;
}

std::vector<pointf2> get_ls_results(const std::vector<pointf2>& points, int n, float lb, float rb, int number) {
	if (points.size() <= 1) return {};
	int m = points.size();
	if (n >= m) return {};
	//return;

	// G^T G A = G^T Y

	Eigen::MatrixXf G(m, n);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			G(i, j) = j == 0 ? 1.0f : G(i, j - 1) * points[i][0];
		}
	}

	Eigen::VectorXf Y(m);
	for (int i = 0; i < m; ++i)
		Y(i) = points[i][1];

	Eigen::VectorXf A = (G.transpose() * G).inverse() * G.transpose() * Y;

	// f(x) = a0 + a_1*g_1(x) + ... + a_n*g_n(x)
	auto f = [&](float x) {
		float ans = 0, xx = 1.0f;
		for (int i = 0; i < n; i++) {
			ans += A(i) * xx; xx *= x;
		}
		return ans;
	};

	std::vector<pointf2> ans;
	for (int i = 0; i < number; i++) {
		float x = lb + (rb - lb) / number * i;
		ans.push_back(pointf2(x, f(x)));
	}
	return ans;
}

std::vector<pointf2> get_ridge_results(const std::vector<pointf2>& points, int n, float lambda, float lb, float rb, int number) {
	if (points.size() <= 1) return {};
	int m = points.size();
	if (n >= m) return {};

	// (G^T G + lambda E) A = G^T Y
	Eigen::MatrixXf G(m, n);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			G(i, j) = j == 0 ? 1.0f : G(i, j - 1) * points[i][0];
		}
	}

	Eigen::VectorXf Y(m);
	for (int i = 0; i < m; ++i)
		Y(i) = points[i][1];

	Eigen::MatrixXf E(n, n);
	E.setIdentity();
	Eigen::VectorXf A = (G.transpose() * G + lambda * E).inverse() * G.transpose() * Y;

	// f(x) = a0 + a_1*g_1(x) + ... + a_n*g_n(x)
	auto f = [&](float x) {
		float ans = 0, xx = 1.0f;
		for (int i = 0; i < n; i++) {
			ans += A(i) * xx; xx *= x;
		}
		return ans;
	};

	std::vector<pointf2> ans;
	for (int i = 0; i < number; i++) {
		float x = lb + (rb - lb) / number * i;
		ans.push_back(pointf2(x, f(x)));
	}
	return ans;
}

PyObject* getNewRBFModel(PyObject* get_new_model, int n) {
	PyObject* args = PyTuple_New(1);
	PyObject* arg1 = PyLong_FromLong(n);
	PyTuple_SetItem(args, 0, arg1);

	PyObject* pRet = PyObject_CallObject(get_new_model, args);

	if (!pRet) {
		spdlog::error("Get new model error");
	}
	else {
		spdlog::error("Get new model success");
	}

	Py_DECREF(args);
	return pRet;
}

void train_rbf(PyObject* train, PyObject* model, const std::vector<pointf2>& points, int& epoch, float& loss) {
	
	PyObject* x_list_py = PyList_New(points.size());
	PyObject* y_list_py = PyList_New(points.size());

	float x_min = 1e9, y_min = 1e9;
	float x_max = -1e9, y_max = -1e9;
	for (size_t i = 0; i < points.size(); i++) {
		float x = points[i][0];
		float y = points[i][1];

		PyList_SetItem(x_list_py, i, PyFloat_FromDouble(x));
		PyList_SetItem(y_list_py, i, PyFloat_FromDouble(y));

		x_min = std::min(x_min, x);
		x_max = std::max(x_max, x);
		y_min = std::min(y_min, y);
		y_max = std::max(y_max, y);
	}

	PyObject* args = PyTuple_New(7);
	PyTuple_SetItem(args, 0, model);
	PyTuple_SetItem(args, 1, x_list_py);
	PyTuple_SetItem(args, 2, y_list_py);
	PyTuple_SetItem(args, 3, PyFloat_FromDouble(x_min));
	PyTuple_SetItem(args, 4, PyFloat_FromDouble(x_max));
	PyTuple_SetItem(args, 5, PyFloat_FromDouble(y_min));
	PyTuple_SetItem(args, 6, PyFloat_FromDouble(y_max));

	PyObject* ret = PyObject_CallObject(train, args);

	if (!ret) {
		spdlog::error("Train model error");
	}
	else {
		//spdlog::error("Train model success");
		int epoch_add;
		float loss_now;
		PyArg_ParseTuple(ret, "i|f", &epoch_add, &loss_now);
		epoch += epoch_add;
		loss = loss_now;
	}

	Py_DECREF(x_list_py);
	Py_DECREF(y_list_py);
	Py_DECREF(args);
	Py_INCREF(model);
}

std::vector<pointf2> get_RBF_results(const std::vector<pointf2>& points, PyObject* test, PyObject* model, float lb, float rb, int number) {
	if (points.size() <= 1) return {};

	float x_min = 1e9, y_min = 1e9;
	float x_max = -1e9, y_max = -1e9;
	for (size_t i = 0; i < points.size(); i++) {
		float x = points[i][0];
		float y = points[i][1];
		x_min = std::min(x_min, x);
		x_max = std::max(x_max, x);
		y_min = std::min(y_min, y);
		y_max = std::max(y_max, y);
	}

	PyObject* x_list_py = PyList_New(number);
	for (int i = 0; i < number; i++) {
		float x = lb + (rb- lb) / number * i;
		PyList_SetItem(x_list_py, i, PyFloat_FromDouble(x));
	}

	//spdlog::info("------------------");

	PyObject* args = PyTuple_New(6);
	PyTuple_SetItem(args, 0, model);
	PyTuple_SetItem(args, 1, x_list_py);
	PyTuple_SetItem(args, 2, PyFloat_FromDouble(x_min));
	PyTuple_SetItem(args, 3, PyFloat_FromDouble(x_max));
	PyTuple_SetItem(args, 4, PyFloat_FromDouble(y_min));
	PyTuple_SetItem(args, 5, PyFloat_FromDouble(y_max));

	PyObject* ret = PyObject_CallObject(test, args);

	std::vector<pointf2> ans;

	if (!ret) {
		spdlog::error("Test error");
	}
	else {
		//spdlog::error("Test success");

		if (!PyList_Check(ret))
			spdlog::error("ret is not list");

		for (int i = 0; i < number; i++) {
			PyObject* py_y = PyList_GetItem(ret, i);
			if (!PyFloat_Check(py_y)) {
				spdlog::error(std::to_string(i) + " is not float");
				continue;
			}
			//if (!PyLong_Check(py_y)) spdlog::error(std::to_string(i) + " is not long");
			//if (!PyNumber_Check(py_y)) spdlog::error(std::to_string(i) + " is not Number");
			float y = PyFloat_AsDouble(py_y);
			if (PyErr_Occurred()) {
				spdlog::error("error at " + std::to_string(i));
			}
			float x = lb + (rb - lb) / number * i;
			ans.push_back(pointf2(x, y));
		}
	}

	Py_DECREF(x_list_py);
	Py_DECREF(args);
	Py_INCREF(model);
	return ans;
}

std::vector<float> parameterize(const std::vector<pointf2>& points, int index) {
	if (points.size() == 0) return {};
	if (points.size() == 1) return { 0.5 };
	if (points.size() == 2) return { 1.0 / 3, 2.0 / 3 };
	
	auto len = [&](pointf2 a, pointf2 b) {
		float x = a[0] - b[0], y = a[1] - b[1];
		return std::sqrt(x * x + y * y);
	};
	auto angle = [&](pointf2 a, pointf2 b, pointf2 c) {
		float x1 = a[0] - b[0], y1 = a[1] - b[1];
		float x2 = c[0] - b[0], y2 = c[1] - b[1];
		float tmp = (x1 * x2 + y1 * y2) / len(a, b) / len(b, c);
		return std::acos(tmp);
	};

	int n = points.size();
	std::vector<float> ans;
	if (index == 0) { // uniform
		float d = 1.0 / (n - 1), x = 0;
		for (int i = 0; i < n; i++) {
			ans.push_back(x);
			x += d;
		}
	}
	else if (index == 1) { // chordal
		float sum = 0;
		for (int i = 0; i + 1 < n; i++) {
			float x = len(points[i], points[i + 1]);
			ans.push_back(sum);
			sum += x;
		}
		ans.push_back(sum);
		for (auto& x : ans) {
			x /= sum;
		}
	}
	else if (index == 2) { // centripetal
		float sum = 0;
		for (int i = 0; i + 1 < n; i++) {
			float x = std::sqrt(len(points[i], points[i + 1]));
			ans.push_back(sum);
			sum += x;
		}
		ans.push_back(sum);
		for (auto& x : ans) {
			x /= sum;
		}
	}
	else if (index == 3) { // Foley
		
		float PI = std::acos(-1);
		auto get = [&](pointf2 a, pointf2 b, pointf2 c) {
			float alpha = std::min(PI - angle(a, b, c), PI / 2);
			return alpha * len(a, b) / (len(a, b) + len(b, c));
		};
		
		float sum = 0;
		for (int i = 0; i + 1 < n; i++) {
			float r = len(points[i], points[i + 1]);
			float f = 0;
			if (i > 0) {
				f += get(points[i - 1], points[i], points[i + 1]);
			}
			if (i < n - 2) {
				f += get(points[i], points[i + 1], points[i + 2]);
			}

			float tmp = r * (1 + 3.0 / 2 * f);
			ans.push_back(sum);
			sum += tmp;
		}
		ans.push_back(sum);
		for (auto& x : ans) {
			x /= sum;
		}
	}
	DEBUG(ans.size() == n);
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

			ImGui::PushStyleColor(ImGuiCol_Text, data->color_poly);
			ImGui::Checkbox("Draw polynomial", &data->opt_enable_polynomial); // lagrange
			ImGui::PopStyleColor();

			ImGui::PushStyleColor(ImGuiCol_Text, data->color_gauss);
			ImGui::Checkbox("Draw Gauss", &data->opt_enable_Gauss); // Gauss
			ImGui::PopStyleColor();
			ImGui::SameLine();
			ImGui::SliderFloat("Sigma", &data->sigma_Gauss, 0.0f, 0.3f);

			ImGui::PushStyleColor(ImGuiCol_Text, data->color_ls);
			ImGui::Checkbox("Draw least squares", &data->opt_enable_least_squares); // least squares
			ImGui::PopStyleColor();
			ImGui::SameLine();
			ImGui::SliderInt("Order 1", &data->n_ls, 0, 10);

			ImGui::PushStyleColor(ImGuiCol_Text, data->color_ridge);
			ImGui::Checkbox("Draw ridge regression", &data->opt_enable_ridge); // ridge
			ImGui::PopStyleColor();
			ImGui::SameLine();
			ImGui::SetNextItemWidth(100);
			ImGui::SliderInt("Order 2", &data->n_ridge, 0, 10);
			ImGui::SameLine();
			ImGui::SetNextItemWidth(100);
			ImGui::SliderFloat("Lambda", &data->lambda_ridge, 0.0f, 0.3f);

			ImGui::PushStyleColor(ImGuiCol_Text, data->color_rbf);
			ImGui::Checkbox("Draw RBF", &data->opt_enable_rbf); // RBF
			ImGui::PopStyleColor();
			ImGui::SameLine();
			if (ImGui::SmallButton("New model")) { 
				data->rbf_loss[0] = data->rbf_loss[1] = -1;
				data->rbf_epoch[0] = data->rbf_epoch[1] = 0;
				data->rbf_is_training = false;
				Py_DECREF(data->py_models[0]);
				Py_DECREF(data->py_models[1]);
				data->py_models[0] = getNewRBFModel(data->py_get_new_model, data->rbf_n);
				data->py_models[1] = getNewRBFModel(data->py_get_new_model, data->rbf_n);
				Py_INCREF(data->py_models[0]);
				Py_INCREF(data->py_models[1]);
			}
			ImGui::SameLine();
			ImGui::SetNextItemWidth(100);
			ImGui::SliderInt("Hidden layer size", &data->rbf_n, 1, 100);
			ImGui::SameLine();
			if (ImGui::SmallButton("Start/Stop training")) {
				data->rbf_is_training ^= 1;
			}
			ImGui::Text(("Model 0: Iteration: " + std::to_string(data->rbf_epoch[0])).c_str());
			ImGui::SameLine();
			ImGui::Text(("Loss: " + std::to_string(data->rbf_loss[0])).c_str());
			ImGui::Text(("Model 1: Iteration: " + std::to_string(data->rbf_epoch[1])).c_str());
			ImGui::SameLine();
			ImGui::Text(("Loss: " + std::to_string(data->rbf_loss[1])).c_str());

			ImGui::Checkbox("Parameterization", &data->opt_enable_params); // params
			ImGui::SameLine();
			ImGui::RadioButton("Uniform", &data->params_indix, 0); // uniform
			ImGui::SameLine();
			ImGui::RadioButton("Chordal", &data->params_indix, 1); // chordal
			ImGui::SameLine();
			ImGui::RadioButton("Centripetal", &data->params_indix, 2); // centripetal
			ImGui::SameLine();
			ImGui::RadioButton("Foley", &data->params_indix, 3); // Foley

			/*ImGui::Text((
				"!! tmp text: data->rbf_n=" + std::to_string(data->rbf_n) 
				+ " data->rbf_is_training=" + std::to_string(data->rbf_is_training)
				+ " data->init=" + std::to_string(data->is_initialize)
				+ " data->debug_val=" + std::to_string(data->debug_val)
			).c_str());*/

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

			// Add first and second point
			if (is_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			{
				//data->points.push_back(mouse_pos_in_canvas);
				//pushPoint(data, mouse_pos_in_canvas);
				data->pushPoint(mouse_pos_in_canvas);
			}

			// Pan (we use a zero mouse threshold when there's no context menu)
			// You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
			const float mouse_threshold_for_pan = data->opt_enable_context_menu ? -1.0f : 0.0f;
			if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan))
			{
				data->scrolling[0] += io.MouseDelta.x;
				data->scrolling[1] += io.MouseDelta.y;
			}

			// Context menu (under default mouse threshold)
			ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
			if (data->opt_enable_context_menu && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && drag_delta.x == 0.0f && drag_delta.y == 0.0f)
				ImGui::OpenPopupContextItem("context");
			if (ImGui::BeginPopup("context"))
			{
				//if (ImGui::MenuItem("Remove one", NULL, false, data->points.size() > 0)) { data->points.resize(data->points.size() - 1); }
				if (ImGui::MenuItem("Remove all", NULL, false, data->points.size() > 0)) { data->clearPoints(); }
				ImGui::EndPopup();
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

			auto norm = [](const std::vector<pointf2>& a, float& x_min, float& x_max, float& y_min, float& y_max) {
				auto ans = a;
				x_min = 1e9, y_min = 1e9;
				x_max = -1e9, y_max = -1e9;
				for (size_t i = 0; i < ans.size(); i++) {
					float x = ans[i][0];
					float y = ans[i][1];

					x_min = std::min(x_min, x);
					x_max = std::max(x_max, x);
					y_min = std::min(y_min, y);
					y_max = std::max(y_max, y);
				}

				for (size_t i = 0; i < ans.size(); i++) {
					float& x = ans[i][0];
					float& y = ans[i][1];
					x = (x - x_min) / (x_max - x_min);
					y = (y - y_min) / (y_max - y_min);
				}
				return ans;
			};

			auto norm_inv = [](const std::vector<pointf2>& a, float x_min, float x_max, float y_min, float y_max) {
				auto ans = a;
				for (size_t i = 0; i < ans.size(); i++) {
					float& x = ans[i][0];
					float& y = ans[i][1];
					x = x * (x_max - x_min) + x_min;
					y = y * (y_max - y_min) + y_min;
				}
				return ans;
			};
				
			if (data->points.size() >= 2) {
				if (!data->opt_enable_params) {

					float x_min, x_max, y_min, y_max;
					auto points = norm(data->points, x_min, x_max, y_min, y_max);

					//const float lb = data->points[0][0] - 5;
					//const float rb = data->points[data->points.size()-1][0] + 5;
					const float lb = 0;
					const float rb = 1;
					const int number = 1000;

					// Draw lines
					if (data->opt_enable_lines) {
						draw(draw_list, norm_inv(points, x_min, x_max, y_min, y_max), origin, data->color_lines);
					}

					if (data->opt_enable_polynomial) {
						std::vector<pointf2> ans = get_poly_results(points, lb, rb, number);
						draw(draw_list, norm_inv(ans, x_min, x_max, y_min, y_max), origin, data->color_poly);
						//draw(draw_list, ans, origin, data->color_poly);
					}

					if (data->opt_enable_Gauss) {
						std::vector<pointf2> ans = get_gauss_results(points, data->sigma_Gauss, lb, rb, number);
						draw(draw_list, norm_inv(ans, x_min, x_max, y_min, y_max), origin, data->color_gauss);
						//draw(draw_list, ans, origin, data->color_gauss);
					}

					if (data->opt_enable_least_squares) {
						std::vector<pointf2> ans = get_ls_results(points, data->n_ls, lb, rb, number);
						draw(draw_list, norm_inv(ans, x_min, x_max, y_min, y_max), origin, data->color_ls);
						//draw(draw_list, ans, origin, data->color_ls);
					}

					if (data->opt_enable_ridge) {
						std::vector<pointf2> ans = get_ridge_results(points, data->n_ridge, data->lambda_ridge, lb, rb, number);
						draw(draw_list, norm_inv(ans, x_min, x_max, y_min, y_max), origin, data->color_ridge);
						//draw(draw_list, ans, origin, data->color_ridge);
					}

					if (data->rbf_is_training) {
						train_rbf(data->py_train, data->py_models[0] , points, data->rbf_epoch[0], data->rbf_loss[0]);
					}

					if (data->opt_enable_rbf) {
						std::vector<pointf2> ans = get_RBF_results(points, data->py_test, data->py_models[0], lb, rb, number);
						draw(draw_list, norm_inv(ans, x_min, x_max, y_min, y_max), origin, data->color_rbf);
						//draw(draw_list, ans, origin, data->color_rbf);
					}
				}
				else {

					float x_min, x_max, y_min, y_max;
					auto points = norm(data->points_input, x_min, x_max, y_min, y_max);
					const float lb = 0;
					const float rb = 1;
					const int number = 1000;

					std::vector<float> t_vec = parameterize(points, data->params_indix);
					std::vector<pointf2> x_list, y_list;
					for (int i = 0; i < t_vec.size(); i++) {
						x_list.push_back(pointf2(t_vec[i], points[i][0]));
						y_list.push_back(pointf2(t_vec[i], points[i][1]));
					}

					auto merge = [&](const std::vector<pointf2>& ans_x, const std::vector<pointf2>& ans_y) {
						std::vector<pointf2> ans;
						for (size_t i = 0; i < ans_x.size(); i++) {
							DEBUG(std::abs(ans_x[i][0] - ans_y[i][0]) < 1e-6);
							ans.push_back(pointf2(ans_x[i][1], ans_y[i][1]));
						//	spdlog::info("(" + std::to_string(ans_x[i][1]) + ", " + std::to_string(ans_y[i][1]) + ")");
						}
						return ans;
					};

					if (data->opt_enable_lines) {
						draw(draw_list, norm_inv(points, x_min, x_max, y_min, y_max), origin, data->color_lines);
					}

					if (data->opt_enable_polynomial) {
						std::vector<pointf2> ans_x = get_poly_results(x_list, lb, rb, number);
						std::vector<pointf2> ans_y = get_poly_results(y_list, lb, rb, number);
						draw(draw_list, norm_inv(merge(ans_x, ans_y), x_min, x_max, y_min, y_max), origin, data->color_poly);
					}

					if (data->opt_enable_Gauss) {
						std::vector<pointf2> ans_x = get_gauss_results(x_list, data->sigma_Gauss, lb, rb, number);
						std::vector<pointf2> ans_y = get_gauss_results(y_list, data->sigma_Gauss, lb, rb, number);
						draw(draw_list, norm_inv(merge(ans_x, ans_y), x_min, x_max, y_min, y_max), origin, data->color_gauss);

					}

					if (data->opt_enable_least_squares) {
						std::vector<pointf2> ans_x = get_ls_results(x_list, data->n_ls, lb, rb, number);
						std::vector<pointf2> ans_y = get_ls_results(y_list, data->n_ls, lb, rb, number);
						draw(draw_list, norm_inv(merge(ans_x, ans_y), x_min, x_max, y_min, y_max), origin, data->color_ls);
					}

					if (data->opt_enable_ridge) {
						std::vector<pointf2> ans_x = get_ridge_results(x_list, data->n_ridge, data->lambda_ridge, lb, rb, number);
						std::vector<pointf2> ans_y = get_ridge_results(y_list, data->n_ridge, data->lambda_ridge, lb, rb, number);
						draw(draw_list, norm_inv(merge(ans_x, ans_y), x_min, x_max, y_min, y_max), origin, data->color_ridge);
					}

					if (data->rbf_is_training) {
						train_rbf(data->py_train, data->py_models[0], x_list, data->rbf_epoch[0], data->rbf_loss[0]);
						train_rbf(data->py_train, data->py_models[1], y_list, data->rbf_epoch[1], data->rbf_loss[1]);
					}

					if (data->opt_enable_rbf) {
						std::vector<pointf2> ans_x = get_RBF_results(x_list, data->py_test, data->py_models[0], lb, rb, number);
						std::vector<pointf2> ans_y = get_RBF_results(y_list, data->py_test, data->py_models[1], lb, rb, number);
						draw(draw_list, norm_inv(merge(ans_x, ans_y), x_min, x_max, y_min, y_max), origin, data->color_rbf);
					}
				}
			}

			// Draw points
			const float point_radius = 5.0f;
			for (int n = 0; n < data->points.size(); n++)
				draw_list->AddCircleFilled(ImVec2(data->points[n][0] + origin.x, data->points[n][1] + origin.y), point_radius, IM_COL32(255, 255, 255, 255));

			draw_list->AddCircleFilled(ImVec2(mouse_pos_in_canvas[0] + origin.x, mouse_pos_in_canvas[1] + origin.y), point_radius, IM_COL32(255, 0, 0, 255));

			draw_list->PopClipRect();
		}

		ImGui::End();
		});
}