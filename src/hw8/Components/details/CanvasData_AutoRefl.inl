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
        Field {TSTR("points"), &Type::points},
        Field {TSTR("points_number"), &Type::points_number, AttrList {
            Attr {TSTR(UMeta::initializer), []()->int{ return { 10 }; }},
        }},
        Field {TSTR("panel_pos"), &Type::panel_pos, AttrList {
            Attr {TSTR(UMeta::initializer), []()->Ubpa::valf2{ return { 50, 30 }; }},
        }},
        Field {TSTR("panel_size"), &Type::panel_size, AttrList {
            Attr {TSTR(UMeta::initializer), []()->Ubpa::valf2{ return { 500, 300 }; }},
        }},
        Field {TSTR("scrolling"), &Type::scrolling, AttrList {
            Attr {TSTR(UMeta::initializer), []()->Ubpa::valf2{ return { 0.f,0.f }; }},
        }},
        Field {TSTR("opt_enable_grid"), &Type::opt_enable_grid, AttrList {
            Attr {TSTR(UMeta::initializer), []()->bool{ return { true }; }},
        }},
        Field {TSTR("opt_enable_context_menu"), &Type::opt_enable_context_menu, AttrList {
            Attr {TSTR(UMeta::initializer), []()->bool{ return { true }; }},
        }},
        Field {TSTR("is_Lloyd_running"), &Type::is_Lloyd_running, AttrList {
            Attr {TSTR(UMeta::initializer), []()->bool{ return { false }; }},
        }},
        Field {TSTR("Lloyd_times"), &Type::Lloyd_times, AttrList {
            Attr {TSTR(UMeta::initializer), []()->int{ return { 1 }; }},
        }},
        Field {TSTR("is_draw_triangle"), &Type::is_draw_triangle, AttrList {
            Attr {TSTR(UMeta::initializer), []()->bool{ return { false }; }},
        }},
        Field {TSTR("is_draw_Voronoi"), &Type::is_draw_Voronoi, AttrList {
            Attr {TSTR(UMeta::initializer), []()->bool{ return { false }; }},
        }},
        Field {TSTR("mouse_point"), &Type::mouse_point, AttrList {
            Attr {TSTR(UMeta::initializer), []()->Ubpa::pointf2{ return { 0, 0 }; }},
        }},
    };
};
