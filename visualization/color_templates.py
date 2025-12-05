from typing import Dict, Tuple, Callable

Color = Tuple[int, int, int]
Opacity = float
VisualProperties = Dict[str, Tuple[Color, Opacity]]

def default_color_template_abc() -> VisualProperties:
    return {
        'BSpline':  ((138, 43, 226), 1.0),
        'Cone':        ((0, 0, 255), 1.0),
        'Cylinder':    ((255, 0, 0), 1.0),
        'Extrusion': ((255, 165, 0), 1.0),
        'Other':   ((128, 128, 128), 1.0),
        'Plane':    ((255, 20, 147), 1.0),
        'Revolution':  ((0, 128, 0), 1.0),
        'Sphere':      ((128, 0, 0), 1.0),
        'Torus':     ((0, 255, 255), 1.0),
        'Void':          ((0, 0, 0), 0.0),
    }

def inside_outside_color_template_abc() -> VisualProperties:
    return {
        'BSpline':  ((138, 43, 226), 1.0),   # purple
        'Cone':     ((0, 128, 0),    1.0),   # green
        'Cylinder': ((255, 165, 0),  1.0),   # orange
        'Plane':    ((255, 20, 147), 1.0),   # deep pink
        'Sphere':   ((128, 0, 0),    1.0),   # maroon
        'Torus':    ((0, 255, 255),  1.0),   # cyan
        'Inside':   ((255, 0, 0),    0.0),   # red
        'Outside':  ((0, 0, 255),    0.0),   # blue
    }

def primitive_template_abc() -> VisualProperties:
    return {
        'Cone':     ((0, 128, 0),    1.0),   # green
        'Cylinder': ((255, 165, 0),  1.0),   # orange
        'Plane':    ((255, 20, 147), 1.0),   # deep pink
        'Sphere':   ((128, 0, 0),    1.0),   # maroon
        'Torus':    ((0, 255, 255),  1.0),   # cyan
        'Inside':   ((255, 0, 0),    0.0),   # red
        'Outside':  ((0, 0, 255),    0.0),   # blue
    }

def edge_color_template_abc() -> VisualProperties:
    return {
        'BSpline':  ((138, 43, 226), 1.0),   # purple
        'Cone':     ((0, 128, 0),    1.0),   # green
        'Cylinder': ((255, 165, 0),  1.0),   # orange
        'Plane':    ((255, 20, 147), 1.0),   # deep pink
        'Sphere':   ((128, 0, 0),    1.0),   # maroon
        'Torus':    ((0, 255, 255),  1.0),   # cyan
        'Edge':     ((255, 255, 0),  1.0),   # yellow
        'Inside':   ((255, 0, 0),    1.0),   # red
        'Outside':  ((0, 0, 255),    0.0),   # blue
    }

def primitive_edge_color_template_abc() -> VisualProperties:
    return {
        'Cone':     ((0, 128, 0),    1.0),   # green
        'Cylinder': ((255, 165, 0),  1.0),   # orange
        'Plane':    ((255, 20, 147), 1.0),   # deep pink
        'Sphere':   ((128, 0, 0),    1.0),   # maroon
        'Torus':    ((0, 255, 255),  1.0),   # cyan
        'Edge':     ((255, 255, 0),  1.0),   # yellow
        'Inside':   ((255, 0, 0),    1.0),   # red
        'Outside':  ((0, 0, 255),    0.0),   # blue
    }

def edge_color_template_abc_sorted() -> VisualProperties:
    return {
        'BSpline':  ((138, 43, 226), 1.0),   # purple
        'Cone':     ((0, 128, 0),    1.0),   # green
        'Cylinder': ((255, 165, 0),  1.0),   # orange
        'Edge':     ((255, 255, 0), 1.0),  # yellow
        'Inside':   ((255, 0, 0), 1.0),  # red
        'Outside':  ((0, 0, 255), 0.0),  # blue
        'Plane':    ((255, 20, 147), 1.0),   # deep pink
        'Sphere':   ((128, 0, 0),    1.0),   # maroon
        'Torus':    ((0, 255, 255),  1.0),   # cyan
    }


def edge_only_color_template_abc() -> VisualProperties:
    return {
        'Surface':   ((128, 0, 0),   1.0),   # maroon
        'Edge':     ((255, 255, 0),  1.0),   # yellow
        'Inside':   ((255, 0, 0),    0.0),   # red
        'Outside':  ((0, 0, 255),    0.0),   # blue
    }

def get_color(template: VisualProperties, class_name: str) -> Color:
    return template.get(class_name, template['Other'])[0]

def get_opacity(template: VisualProperties, class_name: str) -> Opacity:
    return template.get(class_name, template['Other'])[1]

def get_color_dict(template: VisualProperties) -> Dict[str, Color]:
    return {k: v[0] for k, v in template.items()}

def get_opacity_dict(template: VisualProperties) -> Dict[str, Opacity]:
    return {k: v[1] for k, v in template.items()}

def get_class_list(template: VisualProperties) -> list[str]:
    return list(template.keys())

def get_class_to_index_dict(template: VisualProperties) -> Dict[str, int]:
    """Returns a mapping from class name to index (0 to n-1)."""
    return {class_name: idx for idx, class_name in enumerate(template.keys())}

def get_index_to_class_dict(template: VisualProperties) -> Dict[int, str]:
    """Returns the inverse mapping from index to class name."""
    return {idx: class_name for idx, class_name in enumerate(template.keys())}

def get_template_by_name(template_name: str):

    templates = {
        "inside_outside" : 0,
        "edge" : 1,
        "primitive" : 2,
        "primitive_edge" : 3
    }

    code = templates.get(template_name)
    if code is None:
        raise ValueError(f"Unknown template name '{template_name}'")

    match code:
        case 0: return inside_outside_color_template_abc()
        case 1: return edge_color_template_abc()
        case 2: return primitive_template_abc()
        case 3: return primitive_edge_color_template_abc()
        case _: raise RuntimeError(f"Unhandled template code: {code}")


def main():
    template = default_color_template_abc()
    print(get_color(template, 'Cone'))  # (0, 0, 255)
    print(get_opacity(template, 'Void'))  # 0.0
    print(get_color_dict(template).keys())
    print(get_opacity_dict(template))
    pass

if __name__ == "__main__":
    main()

