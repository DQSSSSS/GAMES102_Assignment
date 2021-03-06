// This file is generated by Ubpa::USRefl::AutoRefl

#pragma once

#include <USRefl/USRefl.h>

template<>
struct Ubpa::USRefl::TypeInfo<CanvasData> :
    TypeInfoBase<CanvasData>
{
#ifdef UBPA_USREFL_NOT_USE_NAMEOF
    static constexpr char name[11] = "CanvasData";
#endif
    static constexpr AttrList attrs = {};
    static constexpr FieldList fields = {
        Field {TSTR("csc"), &Type::csc},
        Field {TSTR("points_input"), &Type::points_input},
        Field {TSTR("point_now"), &Type::point_now},
        Field {TSTR("scrolling"), &Type::scrolling, AttrList {
            Attr {TSTR(UMeta::initializer), []()->Ubpa::valf2{ return { 0.f,0.f }; }},
        }},
        Field {TSTR("opt_enable_grid"), &Type::opt_enable_grid, AttrList {
            Attr {TSTR(UMeta::initializer), []()->bool{ return { true }; }},
        }},
        Field {TSTR("opt_enable_context_menu"), &Type::opt_enable_context_menu, AttrList {
            Attr {TSTR(UMeta::initializer), []()->bool{ return { true }; }},
        }},
        Field {TSTR("opt_enable_lines"), &Type::opt_enable_lines, AttrList {
            Attr {TSTR(UMeta::initializer), []()->bool{ return { true }; }},
        }},
        Field {TSTR("color_lines"), &Type::color_lines, AttrList {
            Attr {TSTR(UMeta::initializer), []()->ImU32{ return { IM_COL32(0, 255, 255, 255) }; }},
        }},
        Field {TSTR("is_initialize"), &Type::is_initialize, AttrList {
            Attr {TSTR(UMeta::initializer), []()->bool{ return { false }; }},
        }},
        Field {TSTR("is_drawing_point"), &Type::is_drawing_point, AttrList {
            Attr {TSTR(UMeta::initializer), []()->bool{ return { false }; }},
        }},
        Field {TSTR("is_editing"), &Type::is_editing, AttrList {
            Attr {TSTR(UMeta::initializer), []()->bool{ return { false }; }},
        }},
        Field {TSTR("opt_enable_params"), &Type::opt_enable_params, AttrList {
            Attr {TSTR(UMeta::initializer), []()->bool{ return { false }; }},
        }},
        Field {TSTR("params_indix"), &Type::params_indix, AttrList {
            Attr {TSTR(UMeta::initializer), []()->int{ return { 0 }; }},
        }},
        Field {TSTR("curve_indix"), &Type::curve_indix, AttrList {
            Attr {TSTR(UMeta::initializer), []()->int{ return { 0 }; }},
        }},
        Field {TSTR("initialize"), &Type::initialize},
        Field {TSTR("build_curve"), &Type::build_curve},
        Field {TSTR("found_control_point"), &Type::found_control_point},
        Field {TSTR("move_control_point"), &Type::move_control_point},
        Field {TSTR("get_curve"), &Type::get_curve, AttrList {
            Attr {TSTR(UMeta::default_functions), std::tuple {
                [](Type * __this, float lb, float rb){ return __this->get_curve(std::forward<float>(lb), std::forward<float>(rb)); },
                [](Type * __this, float lb){ return __this->get_curve(std::forward<float>(lb)); },
                [](Type * __this){ return __this->get_curve(); }
            }},
        }},
        Field {TSTR("get_focus_point_id"), &Type::get_focus_point_id},
        Field {TSTR("get_control_points"), &Type::get_control_points},
        Field {TSTR("push_point"), &Type::push_point},
        Field {TSTR("get_all_input_points"), &Type::get_all_input_points},
        Field {TSTR("clear_points"), &Type::clear_points},
        Field {TSTR("del_last_point"), &Type::del_last_point},
    };
};

