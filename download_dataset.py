import kagglehub
from pathlib import Path
import shutil


def download_dataset():
    repo_root = Path(__file__).resolve().parent
    download_dir = repo_root / "dataset_download"

    # KaggleHub requires output_dir to be empty
    download_dir.mkdir(exist_ok=True)

    print(f"Downloading DroneVehicle dataset to: {download_dir}")

    path = kagglehub.dataset_download(
        "brendanalvey/visdrone-dronevehicle", output_dir=str(download_dir)
    )

    print("Download/extraction complete.")
    print(f"Dataset files located at: {path}")

    # Move extracted contents into repo root
    for item in download_dir.iterdir():
        destination = repo_root / item.name

        if destination.exists():
            raise FileExistsError(
                f"Cannot move {item.name}: {destination} already exists."
            )

        shutil.move(str(item), str(destination))

    # Remove now-empty temporary directory
    download_dir.rmdir()

    print(f"Dataset moved into repo root: {repo_root}")


if __name__ == "__main__":
    download_dataset()
