from bt_animation import (
    TreeNode, Render, Tick, SetStatus, Status, Wait,
    ShowOverlay, RenameNode, ColorsDark)


def main():
    root = TreeNode(0, "Root", [
        TreeNode(1, "PauseResumeController RESUMED", [
            TreeNode(2, "PersistentSeq IDX=0", [
                TreeNode(3, "N1"),
                TreeNode(4, "N2"),
                TreeNode(5, "N3")
            ]),
            TreeNode(6, "AlwaysSuccess"),
            TreeNode(7, "Retry", [TreeNode(8, "ON_PAUSE")]),
            TreeNode(9, "Retry", [TreeNode(10, "ON_RESUME")])
        ])
    ])

    actions = [
        SetStatus(0, Status.RUNNING),
        Wait(0.5),

        Tick(0, 1, Status.RUNNING),
        Tick(1, 2, Status.RUNNING),
        Tick(2, 3, Status.SUCCESS),
        RenameNode(2, "PersistentSeq IDX=1"),
        Wait(0.5),
        Tick(0, 1, Status.RUNNING),
        Tick(1, 2, Status.RUNNING),
        Tick(2, 4, Status.RUNNING),
        Wait(0.5),
        Tick(0, 1, Status.RUNNING),
        Tick(1, 2, Status.RUNNING),
        Tick(2, 4, Status.RUNNING),
        Wait(0.5),

        SetStatus(2, Status.IDLE),
        SetStatus(3, Status.IDLE),
        SetStatus(4, Status.IDLE),
        ShowOverlay("Pause service called", 1.5),
        RenameNode(1, "PauseResumeController ON_PAUSE"),
        Tick(0, 1, Status.RUNNING),
        Tick(1, 7, Status.RUNNING),
        Tick(7, 8, Status.FAILURE),
        Wait(0.5),
        Tick(0, 1, Status.RUNNING),
        Tick(1, 7, Status.RUNNING),
        Tick(7, 8, Status.SUCCESS),
        SetStatus(7, Status.SUCCESS),
        Wait(0.5),

        RenameNode(1, "PauseResumeController PAUSED"),
        Tick(0, 1, Status.RUNNING),
        Tick(1, 6, Status.SUCCESS),
        Wait(0.5),
        Tick(0, 1, Status.RUNNING),
        Tick(1, 6, Status.SUCCESS),
        Wait(0.5),
        Tick(0, 1, Status.RUNNING),
        Tick(1, 6, Status.SUCCESS),
        Wait(0.5),

        ShowOverlay("Resume service called", 1.5),
        RenameNode(1, "PauseResumeController ON_RESUME"),
        Tick(0, 1, Status.RUNNING),
        Tick(1, 9, Status.RUNNING),
        Tick(9, 10, Status.SUCCESS),
        SetStatus(9, Status.SUCCESS),
        Wait(0.5),

        RenameNode(1, "PauseResumeController RESUMED"),
        Tick(0, 1, Status.RUNNING),
        Tick(1, 2, Status.RUNNING),
        Tick(2, 4, Status.SUCCESS),
        RenameNode(2, "PersistentSeq IDX=2"),
        Wait(0.5),
        Tick(0, 1, Status.RUNNING),
        Tick(1, 2, Status.RUNNING),
        Tick(2, 5, Status.SUCCESS),
        RenameNode(2, "PersistentSeq IDX=0"),
        SetStatus(2, Status.SUCCESS),
        SetStatus(1, Status.SUCCESS),
        SetStatus(0, Status.SUCCESS),
        Wait(1.0),
    ]

    Render(
        root, actions, canvas_width=1100, canvas_height=500,
        tick_time_s=0.65, colors=ColorsDark).render(filename="bt_animation.mp4")


if __name__ == "__main__":
    main()
