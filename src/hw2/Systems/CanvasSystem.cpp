#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <Python.h>

#include <spdlog/spdlog.h>

using namespace Ubpa;

void draw(ImDrawList* draw_list, std::vector<pointf2>& points, ImVec2 origin, ImU32 color) {
	for (size_t n = 0; n + 1 < points.size(); n++)
		draw_list->AddLine(ImVec2(origin.x + points[n][0], origin.y + points[n][1]), 
			ImVec2(origin.x + points[n + 1][0], origin.y + points[n + 1][1]), color, 2.0f);
}

void drawFunc(ImDrawList* draw_list, std::vector<pointf2>& points, ImVec2 origin, std::function<float(float)> f, ImU32 color) {
	const float x_min = points[0][0] - 5;
	const float x_max = points[points.size() - 1][0] + 5;
	const int number = 1000;
	std::vector<pointf2> ans;
	for (int i = 0; i < number; i++) {
		float x = x_min + (x_max - x_min) / number * i;
		ans.push_back(pointf2(x, f(x)));
	}
	draw(draw_list, ans, origin, color);
}

void drawLines(ImDrawList* draw_list, std::vector<pointf2>& points, ImVec2 origin) {
	draw(draw_list, points, origin, IM_COL32(0, 255, 255, 255));
}

void drawPoly(ImDrawList* draw_list, std::vector<pointf2>& points, ImVec2 origin) {
	if (points.size() <= 1) return;

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

	drawFunc(draw_list, points, origin, f, IM_COL32(255, 0, 255, 255));
}

void drawGauss(ImDrawList* draw_list, std::vector<pointf2>& points, ImVec2 origin, float sigma) {
	if (points.size() <= 1) return;
	
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

	drawFunc(draw_list, points, origin, f, IM_COL32(255, 255, 0, 255));
}

void drawLeastSquares(ImDrawList* draw_list, std::vector<pointf2>& points, ImVec2 origin, int n) {
	if (points.size() <= 1) return;
	int m = points.size();
	if (n >= m) return;
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

	drawFunc(draw_list, points, origin, f, IM_COL32(0, 0, 255, 255));
}

void drawRidge(ImDrawList* draw_list, std::vector<pointf2>& points, ImVec2 origin, int n, float lambda) {
	if (points.size() <= 1) return;
	int m = points.size();
	if (n >= m) return;


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

	drawFunc(draw_list, points, origin, f, IM_COL32(0, 255, 0, 255));
}

std::string test() {

	Py_Initialize();

	//int argc = 1;
	//wchar_t* argv[] = { (wchar_t*)" " };
	//PySys_SetArgv(argc, argv);   //加入argv参数 否则出错

	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('C:/Users/10531/data/Documents/Research/course/games102/homeworks/project/src/hw2/')");

	//PyObject* module_name = PyUnicode_DecodeFSDefault("rbf_net");
	PyObject* pModule = PyImport_ImportModule("rbf_net");

	if (!pModule)
		return "Python get module failed.";

	PyObject* pv = PyObject_GetAttrString(pModule, "_add");
	if (!pv || !PyCallable_Check(pv))
		return "Can't find funftion (_add)";

	PyObject* args = PyTuple_New(2);
	PyObject* arg1 = PyLong_FromLong(4);
	PyObject* arg2 = PyLong_FromLong(3);

	PyTuple_SetItem(args, 0, arg1);
	PyTuple_SetItem(args, 1, arg2);

	PyObject* pRet = PyObject_CallObject(pv, args);
	
	if (pRet) {
		long result = PyLong_AsLong(pRet);
		Py_Finalize();
		std::string ans = std::to_string(result);
		return ans;
	}
	else {
		return "no return";
	}
}

void getNewRBFModel(PyObject* get_new_model, int n) {
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
}

void train_rbf(PyObject* train, std::vector<pointf2>& points, int& epoch, float& loss) {
	
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

	PyObject* args = PyTuple_New(6);
	PyTuple_SetItem(args, 0, x_list_py);
	PyTuple_SetItem(args, 1, y_list_py);
	PyTuple_SetItem(args, 2, PyFloat_FromDouble(x_min));
	PyTuple_SetItem(args, 3, PyFloat_FromDouble(x_max));
	PyTuple_SetItem(args, 4, PyFloat_FromDouble(y_min));
	PyTuple_SetItem(args, 5, PyFloat_FromDouble(y_max));

	PyObject* ret = PyObject_CallObject(train, args);

	if (!ret) {
		spdlog::error("Train model error");
	}
	else {
		spdlog::error("Train model success");
		int epoch_add;
		float loss_now;
		PyArg_ParseTuple(ret, "i|f", &epoch_add, &loss_now);
		epoch += epoch_add;
		loss = loss_now;
	}

	Py_DECREF(x_list_py);
	Py_DECREF(y_list_py);
	Py_DECREF(args);
}


void drawRBF(ImDrawList* draw_list, std::vector<pointf2>& points, ImVec2 origin, PyObject* test) {
	if (points.size() <= 1) return;

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

	const float x_range_min = points[0][0] - 5;
	const float x_range_max = points[points.size() - 1][0] + 5;
	const int number = 1000;
	PyObject* x_list_py = PyList_New(number);
	for (int i = 0; i < number; i++) {
		float x = x_range_min + (x_range_max - x_range_min) / number * i;
		PyList_SetItem(x_list_py, i, PyFloat_FromDouble(x));
	}
	
	PyObject* args = PyTuple_New(5);
	PyTuple_SetItem(args, 0, x_list_py);
	PyTuple_SetItem(args, 1, PyFloat_FromDouble(x_min));
	PyTuple_SetItem(args, 2, PyFloat_FromDouble(x_max));
	PyTuple_SetItem(args, 3, PyFloat_FromDouble(y_min));
	PyTuple_SetItem(args, 4, PyFloat_FromDouble(y_max));

	PyObject* ret = PyObject_CallObject(test, args);

	if (!ret) {
		spdlog::error("Test error");
	}
	else {
		spdlog::error("Test success");

		if (!PyList_Check(ret)) 
			spdlog::error("ret is not list");

		std::vector<pointf2> ans;
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
			float x = x_range_min + (x_range_max - x_range_min) / number * i;
			ans.push_back(pointf2(x, y));
		}
		std::string info;
		for (size_t i = 0; i < ans.size(); i++) {
			std::string tmp = "(" + std::to_string(ans[i][0]) + "," + std::to_string(ans[i][1]) + ") ";
			info += tmp;
		}
		draw(draw_list, ans, origin, IM_COL32(255, 0, 0, 255));
	}

	Py_DECREF(x_list_py);
	Py_DECREF(args);
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

			ImGui::Checkbox("Enable grid", &data->opt_enable_grid);
			ImGui::SameLine();
			ImGui::Checkbox("Enable context menu", &data->opt_enable_context_menu);

			ImGui::Checkbox("Draw lines", &data->opt_enable_lines); // line
			ImGui::Checkbox("Draw polynomial", &data->opt_enable_polynomial); // lagrange

			ImGui::Checkbox("Draw Gauss", &data->opt_enable_Gauss); // Gauss
			ImGui::SameLine();
			ImGui::SliderFloat("Sigma for Gauss fitting", &data->sigma_Gauss, 0.0f, 100.0f);

			ImGui::Checkbox("Draw least squares", &data->opt_enable_least_squares); // least squares
			ImGui::SameLine();
			ImGui::SliderInt("The maxinum order 1", &data->n_ls, 0, 10);

			ImGui::Checkbox("Draw ridge regression", &data->opt_enable_ridge); // ridge
			ImGui::SameLine();
			ImGui::SetNextItemWidth(100);
			ImGui::SliderInt("The maxinum order 2", &data->n_ridge, 0, 10);
			ImGui::SameLine();
			ImGui::SetNextItemWidth(100);
			ImGui::SliderFloat("Lambda for ridge regression", &data->lambda_ridge, 0.0f, 100.0f);

			ImGui::Checkbox("Draw RBF", &data->opt_enable_rbf); // RBF
			if (ImGui::SmallButton("New model")) { 
				data->rbf_loss = 999;
				data->rbf_epoch = 0;
				data->rbf_is_training = false;
				getNewRBFModel(data->py_get_new_model, data->rbf_n);
			}
			ImGui::SameLine();
			ImGui::SliderInt("Hidden layer size", &data->rbf_n, 1, 100);
			if (ImGui::SmallButton("Start/Stop training")) {
				data->rbf_is_training ^= 1;
			}
			ImGui::SameLine();
			ImGui::Text(("Now epoch: " + std::to_string(data->rbf_epoch)).c_str());
			ImGui::SameLine();
			ImGui::Text(("Loss: " + std::to_string(data->rbf_loss)).c_str());

			/*ImGui::Text((
				"!! tmp text: data->rbf_n=" + std::to_string(data->rbf_n) 
				+ " data->rbf_is_training=" + std::to_string(data->rbf_is_training)
				+ " data->init=" + std::to_string(data->is_initialize)
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
				if (ImGui::MenuItem("Remove all", NULL, false, data->points.size() > 0)) { data->points.clear(); }
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

			// Draw lines
			if (data->opt_enable_lines) {
				drawLines(draw_list, data->points, origin);
			}

			if (data->opt_enable_polynomial) {
				drawPoly(draw_list, data->points, origin);
			}

			if (data->opt_enable_Gauss) {
				drawGauss(draw_list, data->points, origin, data->sigma_Gauss);
			}

			if (data->opt_enable_least_squares) {
				drawLeastSquares(draw_list, data->points, origin, data->n_ls);
			}

			if (data->opt_enable_ridge) {
				drawRidge(draw_list, data->points, origin, data->n_ridge, data->lambda_ridge);
			}

			if (data->rbf_is_training) {
				train_rbf(data->py_train, data->points, data->rbf_epoch, data->rbf_loss);
			}

			if (data->opt_enable_rbf) {
				drawRBF(draw_list, data->points, origin, data->py_test);
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