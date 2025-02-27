
colors = [
    "#0072BD",
    "#D95319",
    "#77AC30",
    "#7E2F8E",
    "#4DBEEE",
    "#EDB120",
    "#A2142F",
    "#77AC30",
    "#4DBEEE",
    "#A2142F",
]

def darken_hex_color(hex_color, percentage=20):
    """Darken the given hex color by the specified percentage."""
    # Convert hex color to RGB tuple
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    # Darken each RGB component by the given percentage
    darkened_rgb = tuple(max(0, int(c * (1 - percentage / 100))) for c in rgb)

    # Convert RGB tuple back to hex color
    darkened_hex = "#{:02X}{:02X}{:02X}".format(*darkened_rgb)

    return darkened_hex


def lighten_hex_color(hex_color, percentage=40):
    """Lighten the given hex color by the specified percentage."""
    # Convert hex color to RGB tuple
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    # Lighten each RGB component by the given percentage
    lightened_rgb = tuple(
        min(255, int(c + (255 - c) * (percentage / 100))) for c in rgb
    )

    # Convert RGB tuple back to hex color
    lightened_hex = "#{:02X}{:02X}{:02X}".format(*lightened_rgb)

    return lightened_hex

darker_colors = [darken_hex_color(color) for color in colors]
ligther_colors = [lighten_hex_color(color) for color in colors]