import queue
import re
from enum import Enum
from queue import LifoQueue
from typing import List


def args_to_kwargs(defaults, *args, **kwargs):
    """
    Converts lists of unnamed and named params to only named params, using default values where specified.
    :param defaults: List of ('param_name', default_value)
    :param args:
    :param kwargs:
    :return:
    """
    # the kwargs we will actually pass to the function
    out_kwargs = dict()
    arg_i = 0

    # link every unnamed arg to its name, adopt default value if None
    for value in args:
        try:
            arg_name, default_value = defaults[arg_i]
        except IndexError:
            continue
        if value is None:
            value = default_value
        out_kwargs[arg_name] = value
        arg_i += 1

    # get kwargs for any arg we have not yet used
    for arg_name, default_value in defaults[arg_i:]:
        out_kwargs[arg_name] = kwargs.pop(arg_name, default_value)

    # check for repeated params of values whose default has not been given
    for key, value in kwargs.items():
        if key in defaults:
            raise TypeError(f"Received param '{key}' twice (named and unnamed).")

    # add leftover params
    out_kwargs.update(kwargs)

    return out_kwargs


def default(variable, value):
    if variable is None:
        return value
    return variable


def flatten(list_of_lists: List[List]):
    return [elem for sublist in list_of_lists for elem in sublist]


def print_format_table():
    """
    prints table of formatted text format options
    """
    for style in range(108):
        for fg in range(30, 38):
            s1 = ""
            for bg in range(40, 48):
                format = ";".join([str(style), str(fg), str(bg)])
                s1 += f"\x1b[{format}m {format} \x1b[0m"
            print(s1)
        print('\n')


class TextStyle(Enum):
    Reset = 0
    Bold = 1
    Dim = 2
    Italic = 3
    Underline = 4
    SlowBlink = 5
    FastBlink = 6
    Invert = 7
    Hidden = 8
    Strike = 9
    DefaultFont = 10
    AltFont1 = 11
    Gothic = 20
    DoubleUnderline = 21
    StandardIntensity = 22
    NoItalic = 23
    NoUnderline = 24
    NoBlink = 25
    ProportionalSpacing = 26
    NotReversed = 27
    Reveal = 28
    NotStrike = 29
    CustomForeColor = 38
    DefaultForeColor = 39
    CustomBackColor = 48
    DefaultBackColor = 49
    Framed = 51
    Encircled = 52
    Overlined = 53
    CustomUnderColor = 58
    DefaultUnderColor = 59


class TextForeColor(Enum):
    Black = 30
    Red = 31
    Green = 32
    Yellow = 33
    Blue = 34
    Magenta = 35
    Cyan = 36
    White = 37
    BrightRed = 90
    BrightBlack = 91
    BrightGreen = 92
    BrightYellow = 93
    BrightBlue = 94
    BrightMagenta = 95
    BrightCyan = 96
    BrightWhite = 97


class TextBackColor(Enum):
    Reset = 0
    Black = 40
    Red = 41
    Green = 42
    Yellow = 43
    Blue = 44
    Magenta = 45
    Cyan = 46
    White = 47
    BrightRed = 100
    BrightBlack = 101
    BrightGreen = 102
    BrightYellow = 103
    BrightBlue = 104
    BrightMagenta = 105
    BrightCyan = 106
    BrightWhite = 107


styles = {
    'r': TextStyle.Reset.value,
    'b': TextStyle.Bold.value,
    'd': TextStyle.Dim.value,
    'i': TextStyle.Italic.value,
    'u': TextStyle.Underline.value,
    'F': TextStyle.SlowBlink.value,
    'f': TextStyle.FastBlink.value,
    'I': TextStyle.Invert.value,
    'h': TextStyle.Hidden.value,
    's': TextStyle.Strike.value,
    'D': TextStyle.DefaultFont.value,
    'a': TextStyle.AltFont1.value,
    'g': TextStyle.Gothic.value,
    'U': TextStyle.DoubleUnderline.value,
    'x': TextStyle.StandardIntensity.value,
    'î': TextStyle.NoItalic.value,
    'û': TextStyle.NoUnderline.value,
    'B': TextStyle.NoBlink.value,
    'p': TextStyle.ProportionalSpacing.value,
    'R': TextStyle.NotReversed.value,
    'H': TextStyle.Reveal.value,
    'S': TextStyle.NotStrike.value,
    'm': TextStyle.Framed.value,
    'e': TextStyle.Encircled.value,
    'o': TextStyle.Overlined.value,
}
text_colors = {
    'k': TextForeColor.Black.value,
    'r': TextForeColor.Red.value,
    'g': TextForeColor.Green.value,
    'y': TextForeColor.Yellow.value,
    'b': TextForeColor.Blue.value,
    'm': TextForeColor.Magenta.value,
    'c': TextForeColor.Cyan.value,
    'w': TextForeColor.White.value,
    'K': TextForeColor.BrightRed.value,
    'R': TextForeColor.BrightBlack.value,
    'G': TextForeColor.BrightGreen.value,
    'Y': TextForeColor.BrightYellow.value,
    'B': TextForeColor.BrightBlue.value,
    'M': TextForeColor.BrightMagenta.value,
    'C': TextForeColor.BrightCyan.value,
    'W': TextForeColor.BrightWhite.value,
}
background_colors = {
    'k': TextBackColor.Black.value,
    'r': TextBackColor.Red.value,
    'g': TextBackColor.Green.value,
    'y': TextBackColor.Yellow.value,
    'b': TextBackColor.Blue.value,
    'm': TextBackColor.Magenta.value,
    'c': TextBackColor.Cyan.value,
    'w': TextBackColor.White.value,
    'K': TextBackColor.BrightRed.value,
    'R': TextBackColor.BrightBlack.value,
    'G': TextBackColor.BrightGreen.value,
    'Y': TextBackColor.BrightYellow.value,
    'B': TextBackColor.BrightBlue.value,
    'M': TextBackColor.BrightMagenta.value,
    'C': TextBackColor.BrightCyan.value,
    'W': TextBackColor.BrightWhite.value,
}


def txt(text):
    """
    Quick formatting of text style and foreground and background colors. Adding escape sequences.
    Format is '%FBS', where F stands for foreground color, B is background color and S is text style.
    Example: "Hello %rkbWorld" will print "Hello " in standard text and "World" as (r)ed over blac(k) background, in (b)old.

    A space in place of a text parameter will be ignored and the default will be used instead.
    Example: "Hello % kbWorld" will print as before a standard "Hello " followed by "World" in standard text color
    over a black background, in bold.

    The last text format will be used unless reset. The dot '.' can be used to retrieve the format used before the last.
    Example: "Hello %rkbWorld%.k., how are you?" will print as before a standard "Hello " followed by "World" as
    red text color over a black background, in bold.
    At that point the previously used (in this case, the standard) foreground color and style (first and last dots)
    will be in use, and only the black background will show for the remaining string.
    :param text:
    :return:
    """
    pattern = r"(?<!%)%([^%]{3})"
    match = re.search(pattern, text)

    default_style = TextStyle.Reset.value
    default_fore = ""
    default_back = ""

    style_code = default_style
    back_code = default_back
    fore_code = default_fore
    # default_style = "\x1b[6;30;42m"
    text = "\x1b[0m" + text

    style_lifo = LifoQueue()
    fore_lifo = LifoQueue()
    back_lifo = LifoQueue()
    msg = default_style
    while match:
        code = match.group(1)
        fore, back, style = code
        if fore == ".":
            try:
                fore_code = fore_lifo.get(block=False)
            except queue.Empty:
                fore_code = default_fore
        else:
            if fore != " ":
                fore_lifo.put(fore_code)
                fore_code = text_colors[fore]

        if style == ".":
            try:
                style_code = style_lifo.get(block=False)
            except queue.Empty:
                style_code = default_style
        else:
            if style != " ":
                style_lifo.put(style_code)
                style_code = f"{styles[style]};"

        if back == ".":
            try:
                back_code = back_lifo.get(block=False)
            except queue.Empty:
                back_code = default_back
        else:
            if back != " ":
                back_lifo.put(back_code)
                back_code = f";{background_colors[back]}"

        escape_code = f"\x1b[{style_code}{fore_code}{back_code}m"

        text = re.sub(pattern, escape_code, text, count=1)
        match = re.search(pattern, text)
    text = re.sub(r"%%", "%", text)
    return text + "\x1b[0m"


def flatten_iter(obj):
    """
    Iteratively flatten a list of lists [of lists ...] into a list of items
    :param obj:
    :return:
    """
    result = []
    if isinstance(obj, list):
        for x in obj:
            flat = flatten_iter(x)
            if isinstance(flat, list):
                result.extend(flat)
            else:
                result.append(flat)
        return result
    return obj


if __name__ == "__main__":
    print(txt("Hello %r  Friend!%.  how are you? What '%%' are you at? %by Actually,"))
    print(f"\x1b[{TextForeColor.Cyan.value}mHOla")
    print(f"\x1b[{TextStyle.Bold.value};{TextForeColor.Cyan.value}mHOla")
    print_format_table()