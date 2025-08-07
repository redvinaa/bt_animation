import os
from typing import List, Dict, Union
import cv2
import numpy as np
import networkx as nx
import time
import enum


class Status(enum.Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


Colors = {
    Status.IDLE: (240, 240, 240),
    Status.RUNNING: (0, 255, 255),
    Status.SUCCESS: (0, 255, 0),
    Status.FAILURE: (0, 0, 255),
    "ball": (0, 0, 0),
    "background": (255, 255, 255)
}


class TreeNode:
    def __init__(self, node_id: int, name: str, children: List = []):
        self.node_id = node_id
        self.name = name
        self.children = children or []


# --- Actions ---
class Action:
    pass


class Tick(Action):
    def __init__(self, from_id: int, to_id: int, status: Status):
        self.from_id = from_id
        self.to_id = to_id
        self.status = status


class SetStatus(Action):
    def __init__(self, node_id: int, status: Status):
        self.node_id = node_id
        self.status = status


class Wait:
    def __init__(self, duration):
        self.duration = duration


class ShowOverlay:
    def __init__(self, text: str, duration: float):
        self.text = text
        self.duration = duration


class TickParallel(Action):
    def __init__(self, actions: List):
        self.actions = actions


class RenameNode(Action):
    def __init__(self, node_id: int, new_name: str):
        self.node_id = node_id
        self.new_name = new_name


def compute_intersection_with_rect(start, end, width, height):
    dx, dy = end - start
    if dx == 0 and dy == 0:
        return start  # Avoid division by zero for zero-length lines

    scale_x = (width / 2) / abs(dx) if dx != 0 else np.inf
    scale_y = (height / 2) / abs(dy) if dy != 0 else np.inf

    scale = min(scale_x, scale_y)
    return start + scale * np.array([dx, dy])


# --- Renderer main class ---
class Render:
    def __init__(
            self, root: TreeNode, actions: List, *,
            contour_width: int = 1,
            tick_ball_radius: int = 4,
            tick_time_s: float = 0.4,
            canvas_width: int = 800,
            canvas_height: int = 600,
            text_scale: float = 0.7,
            text_thickness: int = 1,
            padding_x: int = 10,  # horizontal padding inside the node box
            padding_y: int = 10):  # vertical padding inside the node box
        self.root = root
        self.actions = actions

        self.contour_width = contour_width
        self.tick_ball_radius = tick_ball_radius
        self.tick_time_s = tick_time_s
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.padding_x = padding_x
        self.padding_y = padding_y

        self.parent: Dict[int, Union[int, None]] = {}
        self.node_names: Dict[int, str] = {}

        def walk(n: TreeNode):
            self.node_names[n.node_id] = n.name
            for c in n.children:
                self.parent[c.node_id] = n.node_id
                walk(c)
        walk(root)
        self.parent[root.node_id] = None

        self.nodes = list(self.parent.keys())
        self.status = {n: Status.IDLE for n in self.nodes}

        # Layout
        G = nx.DiGraph([(p, c) for c, p in self.parent.items() if p is not None])
        try:
            pos = nx.nx_agraph.graphviz_layout(
                G, prog="dot", args="-Grankdir=TB -Gnodesep=10.5 -Granksep=1.3")
        except Exception as e:
            print(f"Graphviz layout failed: {e}, using spring layout instead.")
            pos = nx.spring_layout(G, seed=42, scale=400, center=(400, 300))

        xs = np.array([v[0] for v in pos.values()])
        ys = np.array([v[1] for v in pos.values()])
        minx, maxx = xs.min(), xs.max()
        miny, maxy = ys.min(), ys.max()

        pad_x = 100
        pad_y = 100
        usable_width = self.canvas_width - 2 * pad_x
        usable_height = self.canvas_height - 2 * pad_y

        self.pos = {
            n: (
                int((pos[n][0] - minx) / (maxx - minx) * usable_width + pad_x),
                int((1.0 - (pos[n][1] - miny) / (maxy - miny)) * usable_height + pad_y)
            ) for n in self.nodes
        }

        self.node_size = {}
        for n in self.nodes:
            label = self.node_names[n]
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness)
            w = text_w + 2 * self.padding_x
            h = text_h + 2 * self.padding_y
            self.node_size[n] = (w, h)

    def draw_rounded_rect(self, frame, center, size, fill_color, contour_color, radius=10):
        x, y = center
        w, h = size
        half_w, half_h = w // 2, h // 2
        tl = (x - half_w, y - half_h)
        br = (x + half_w, y + half_h)
        tr = (br[0], tl[1])
        bl = (tl[0], br[1])

        corner_centers = {
            'tl': (tl[0] + radius, tl[1] + radius),
            'tr': (tr[0] - radius, tr[1] + radius),
            'bl': (bl[0] + radius, bl[1] - radius),
            'br': (br[0] - radius, br[1] - radius),
        }

        cv2.rectangle(
            frame, (tl[0] + radius, tl[1] + radius), (br[0] - radius, br[1] - radius),
            fill_color, -1)
        cv2.rectangle(
            frame, (tl[0], tl[1] + radius), (tl[0] + radius, br[1] - radius), fill_color, -1)
        cv2.rectangle(
            frame, (br[0] - radius, tl[1] + radius), (br[0], br[1] - radius), fill_color, -1)
        cv2.rectangle(
            frame, (tl[0] + radius, tl[1]), (br[0] - radius, tl[1] + radius), fill_color, -1)
        cv2.rectangle(
            frame, (tl[0] + radius, br[1] - radius), (br[0] - radius, br[1]), fill_color, -1)
        for c in corner_centers.values():
            cv2.circle(frame, c, radius, fill_color, -1, lineType=cv2.LINE_AA)

        # Bottom highlight line (inside)
        highlight_color = list(fill_color)
        for i in range(3):
            highlight_color[i] = min(255, int(highlight_color[i] - 50))
        highlight_thickness = 2
        inset_offset = radius // 2
        cv2.line(frame,
                 (tl[0] + inset_offset, br[1] - inset_offset),
                 (br[0] - inset_offset, br[1] - inset_offset),
                 highlight_color, highlight_thickness, cv2.LINE_AA)

        # Contour
        if contour_color is not None and self.contour_width > 0:
            cv2.ellipse(
                frame, corner_centers['tl'], (radius, radius),
                180, 0, 90, contour_color, self.contour_width, cv2.LINE_AA)
            cv2.ellipse(
                frame, corner_centers['tr'], (radius, radius),
                270, 0, 90, contour_color, self.contour_width, cv2.LINE_AA)
            cv2.ellipse(
                frame, corner_centers['br'], (radius, radius),
                0, 0, 90, contour_color, self.contour_width, cv2.LINE_AA)
            cv2.ellipse(
                frame, corner_centers['bl'], (radius, radius),
                90, 0, 90, contour_color, self.contour_width, cv2.LINE_AA)

            cv2.line(
                frame, (corner_centers['tl'][0], tl[1]), (corner_centers['tr'][0], tr[1]),
                contour_color, self.contour_width, cv2.LINE_AA)
            cv2.line(
                frame, (corner_centers['bl'][0], bl[1]), (corner_centers['br'][0], br[1]),
                contour_color, self.contour_width, cv2.LINE_AA)
            cv2.line(
                frame, (tl[0], corner_centers['tl'][1]), (bl[0], corner_centers['bl'][1]),
                contour_color, self.contour_width, cv2.LINE_AA)
            cv2.line(
                frame, (tr[0], corner_centers['tr'][1]), (br[0], corner_centers['br'][1]),
                contour_color, self.contour_width, cv2.LINE_AA)

    def draw_ball(self, frame, position):
        pos_int = tuple(position.astype(int))
        cv2.circle(frame, pos_int, self.tick_ball_radius, Colors["ball"], -1)
        cv2.circle(frame, pos_int, self.tick_ball_radius, (0, 0, 0), self.contour_width)

    def draw_tree(self, frame):
        frame.fill(255)
        for c, p in self.parent.items():
            if p is not None:
                cv2.line(frame, self.pos[p], self.pos[c], (0, 0, 0), self.contour_width)

        for n, center in self.pos.items():
            color = Colors[self.status[n]]
            size = self.node_size[n]
            self.draw_rounded_rect(
                frame, center, size, fill_color=color, contour_color=(0, 0, 0), radius=2)

            label = self.node_names[n]
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness)
            text_pos = (center[0] - text_w // 2, center[1] + text_h // 2)
            cv2.putText(
                frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, (0, 0, 0),
                self.text_thickness, lineType=cv2.LINE_AA)

    def render(self, filename: str | None = None):
        frame = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
        win = "BT Animation"
        cv2.namedWindow(win)
        fps = 60

        if filename:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                filename, fourcc, fps, (self.canvas_width, self.canvas_height))
        else:
            writer = None

        def render_frame(img):
            cv2.imshow(win, img)
            if writer:
                writer.write(img.copy())
            if cv2.waitKey(1) == 27:
                raise KeyboardInterrupt

        try:
            for act in self.actions:
                if isinstance(act, SetStatus):
                    self.status[act.node_id] = act.status
                    self.draw_tree(frame)
                    render_frame(frame)
                    continue

                if isinstance(act, Tick):
                    act = TickParallel([act])

                if isinstance(act, Wait):
                    steps = int(act.duration * fps)
                    for _ in range(steps):
                        frame[:] = 255
                        self.draw_tree(frame)
                        render_frame(frame)
                        time.sleep(1 / fps)

                if isinstance(act, ShowOverlay):
                    total_steps = int(act.duration * fps)
                    fade_steps = int(0.1 * total_steps)
                    show_steps = total_steps - 2 * fade_steps
                    base = np.full_like(frame, Colors["background"], dtype=np.uint8)

                    for i in range(fade_steps):
                        self.draw_tree(base)
                        blur = cv2.GaussianBlur(base, (15, 15), 0)
                        alpha = i / fade_steps
                        blended = cv2.addWeighted(base, 1 - alpha, blur, alpha, 0)
                        render_frame(blended)
                        time.sleep(1 / fps)

                    for _ in range(show_steps):
                        self.draw_tree(base)
                        blur = cv2.GaussianBlur(base, (15, 15), 0)
                        cv2.putText(
                            blur, act.text, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, self.text_scale,
                            (0, 0, 0), self.contour_width, lineType=cv2.LINE_AA)
                        render_frame(blur)
                        time.sleep(1 / fps)

                    for i in range(fade_steps):
                        self.draw_tree(base)
                        blur = cv2.GaussianBlur(base, (15, 15), 0)
                        alpha = 1 - (i / fade_steps)
                        blended = cv2.addWeighted(base, 1 - alpha, blur, alpha, 0)
                        render_frame(blended)
                        time.sleep(1 / fps)

                if isinstance(act, TickParallel):
                    positions = []
                    for tick in act.actions:
                        if self.parent.get(tick.to_id) != tick.from_id:
                            raise ValueError(f"{tick.from_id} is not parent of {tick.to_id}")
                        start = np.array(self.pos[tick.from_id])
                        end = np.array(self.pos[tick.to_id])
                        from_size = self.node_size[tick.from_id]
                        to_size = self.node_size[tick.to_id]
                        p_start = compute_intersection_with_rect(start, end, *from_size)
                        p_end = compute_intersection_with_rect(end, start, *to_size)
                        positions.append((p_start, p_end, tick.to_id, tick.status))

                    steps = int(self.tick_time_s * fps)
                    for i in range(steps + 1):
                        self.draw_tree(frame)
                        t = i / steps
                        for p_start, p_end, _, _ in positions:
                            pos = p_start * (1 - t) + p_end * t
                            self.draw_ball(frame, pos)
                        render_frame(frame)
                        time.sleep(self.tick_time_s / steps)

                    for _, _, node_id, status in positions:
                        self.status[node_id] = status

                    self.draw_tree(frame)
                    render_frame(frame)

                if isinstance(act, RenameNode):
                    if act.node_id not in self.node_names:
                        raise ValueError(f"Node {act.node_id} does not exist")
                    self.node_names[act.node_id] = act.new_name
                    (text_w, text_h), _ = cv2.getTextSize(
                        act.new_name, cv2.FONT_HERSHEY_SIMPLEX,
                        self.text_scale, self.text_thickness)
                    w = text_w + 2 * self.padding_x
                    h = text_h + 2 * self.padding_y
                    self.node_size[act.node_id] = (w, h)
                    self.draw_tree(frame)
                    render_frame(frame)

        except KeyboardInterrupt:
            print("Interrupted")
        finally:
            if writer and filename is not None:
                writer.release()
                full_path = os.path.abspath(filename)
                print(f"Video saved to {full_path}")
            cv2.destroyAllWindows()


# --- Usage Example ---
def main():
    # Simple tree with one root and two children
    root = TreeNode(0, "Root", [
        TreeNode(1, "Sequence", [
            TreeNode(2, "Action1"),
            TreeNode(3, "Action2")
        ])
    ])

    actions = [
        # Set initial status
        SetStatus(0, Status.RUNNING),
        Wait(0.5),

        # First tick: Action1 runs and succeeds
        Tick(0, 1, Status.RUNNING),
        Tick(1, 2, Status.SUCCESS),
        RenameNode(2, "Action1 DONE"),
        Wait(0.5),

        # Second tick: Action2 runs and fails
        Tick(0, 1, Status.RUNNING),
        Tick(1, 3, Status.FAILURE),
        ShowOverlay("Action2 failed!", 1.0),
        RenameNode(3, "Action2 FAILED"),
        Wait(0.5),

        # Set final statuses
        SetStatus(1, Status.FAILURE),
        SetStatus(0, Status.FAILURE),
        Wait(1.0),
    ]

    Render(root, actions).render("simple_bt_example.mp4")


if __name__ == "__main__":
    main()
