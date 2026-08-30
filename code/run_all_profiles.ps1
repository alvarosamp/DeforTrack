param(
    [string]$Python = "D:\Datasets\Defortrack\gpu_eval_env\Scripts\python.exe",
    [string]$Output = "C:\Users\vish8\Documents\Codex\2026-08-29\fa-a-o-seguinte-ja-te\outputs\standardized_profile_892_v2.csv"
)

$ErrorActionPreference = "Stop"
$script = "C:\Users\vish8\Documents\Codex\2026-08-29\fa-a-o-seguinte-ja-te\code\profile_model_local.py"
$images = "D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3\valid\images"

$completed = @{}
if (Test-Path -LiteralPath $Output) {
    foreach ($row in (Import-Csv -LiteralPath $Output)) {
        if ($completed.ContainsKey($row.model)) {
            throw "Duplicate model in existing profile output: $($row.model)"
        }
        $completed[$row.model] = $true
    }
    Write-Host "Resuming profile run with $($completed.Count) completed models."
}

$models = @(
    @{ Family="yolo"; Name="YOLOv8n"; Path="C:\Users\vish8\Documents\Codex\2026-07-15\me\outputs\cross_dataset\models_all_yolo\YOLOv8n.pt" },
    @{ Family="yolo"; Name="YOLOv8s"; Path="C:\Users\vish8\Documents\Codex\2026-07-15\me\outputs\cross_dataset\models_all_yolo\YOLOv8s.pt" },
    @{ Family="yolo"; Name="YOLOv8m"; Path="C:\Users\vish8\Documents\Codex\2026-07-15\me\outputs\cross_dataset\models_all_yolo\YOLOv8m.pt" },
    @{ Family="yolo"; Name="YOLOv8l"; Path="C:\Users\vish8\Documents\Codex\2026-07-15\me\outputs\cross_dataset\models_all_yolo\YOLOv8l.pt" },
    @{ Family="yolo"; Name="YOLOv8x"; Path="C:\Users\vish8\Documents\Codex\2026-07-15\me\outputs\cross_dataset\models_all_yolo\YOLOv8x.pt" },
    @{ Family="yolo"; Name="YOLOv11n"; Path="C:\Users\vish8\Documents\Codex\2026-07-15\me\outputs\cross_dataset\models_all_yolo\YOLOv11n.pt" },
    @{ Family="yolo"; Name="YOLOv11s"; Path="C:\Users\vish8\Documents\Codex\2026-07-15\me\outputs\cross_dataset\models_all_yolo\YOLOv11s.pt" },
    @{ Family="yolo"; Name="YOLOv11m"; Path="C:\Users\vish8\Documents\Codex\2026-07-15\me\outputs\cross_dataset\models_all_yolo\YOLOv11m.pt" },
    @{ Family="yolo"; Name="YOLOv11l"; Path="C:\Users\vish8\Documents\Codex\2026-07-15\me\outputs\cross_dataset\models_all_yolo\YOLOv11l.pt" },
    @{ Family="yolo"; Name="YOLOv11x"; Path="C:\Users\vish8\Documents\Codex\2026-07-15\me\outputs\cross_dataset\models_all_yolo\YOLOv11x.pt" },
    @{ Family="detectron"; Name="mask_rcnn_R_101_C4_3x"; Path="D:\Datasets\Defortrack\detectron_eval_models\mask_rcnn_R_101_C4_3x.pth" },
    @{ Family="detectron"; Name="mask_rcnn_R_101_DC5_3x"; Path="D:\Datasets\Defortrack\detectron_eval_models\mask_rcnn_R_101_DC5_3x.pth" },
    @{ Family="detectron"; Name="mask_rcnn_R_101_FPN_3x"; Path="D:\Datasets\Defortrack\detectron_eval_models\mask_rcnn_R_101_FPN_3x.pth" },
    @{ Family="detectron"; Name="mask_rcnn_R_50_C4_1x"; Path="D:\Datasets\Defortrack\detectron_eval_models\mask_rcnn_R_50_C4_1x.pth" },
    @{ Family="detectron"; Name="mask_rcnn_R_50_C4_3x"; Path="D:\Datasets\Defortrack\detectron_eval_models\mask_rcnn_R_50_C4_3x.pth" },
    @{ Family="detectron"; Name="mask_rcnn_R_50_DC5_1x"; Path="D:\Datasets\Defortrack\detectron_eval_models\mask_rcnn_R_50_DC5_1x.pth" },
    @{ Family="unet"; Name="U-Net"; Path="D:\Datasets\Defortrack\unet_instance_segmentation_best.keras" }
)

foreach ($model in $models) {
    if ($completed.ContainsKey($model.Name)) {
        Write-Host "Skipping completed model $($model.Name)."
        continue
    }
    Write-Host "Profiling $($model.Name) on all 892 images..."
    & $Python $script --family $model.Family --model-name $model.Name --model-path $model.Path --images $images --output $Output --warmup 892 --repetitions 3
    if ($LASTEXITCODE -ne 0) {
        throw "Profiling failed for $($model.Name) with exit code $LASTEXITCODE"
    }
}

Write-Host "Completed standardized profiles: $Output"
