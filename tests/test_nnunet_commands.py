from mamaguardnet.nnunet.commands import build_train_command


def test_train_command_selects_resenc_plans_and_single_gpu_smoke_trainer():
    command = build_train_command({
        "dataset": 501,
        "configuration": "3d_fullres",
        "fold": 0,
        "augmentation": "kinetics",
        "trainer": "nnUNetTrainerMAMAGuardSmoke",
        "plans": "nnUNetResEncUNetLPlans",
        "num_gpus": 1,
        "save_probabilities": False,
    })
    assert command == [
        "nnUNetv2_train", "501", "3d_fullres", "0",
        "-tr", "nnUNetTrainerMAMAGuardSmoke",
        "-num_gpus", "1",
        "-p", "nnUNetResEncUNetLPlans",
    ]
