import os
import json
import cv2
import shutil


def download(video_path, ytb_id, proxy=None):
    """
    Download video from YouTube
    :param video_path: Path to save the video
    :param ytb_id: YouTube video ID
    :param proxy: Proxy server URL if required
    """
    proxy_cmd = f"--proxy {proxy}" if proxy else ""
    
    if not os.path.exists(video_path):
        cookies_file = os.path.expanduser("~/cookies.txt")
        download_cmd = " ".join([
        "yt-dlp",
         proxy_cmd,
        "-f \"bestvideo[ext=mp4]\"",
        "--skip-unavailable-fragments",
        "--merge-output-format mp4",
        f"https://www.youtube.com/watch?v={ytb_id}",
        f"--output \"{video_path}\"",
        f"--cookies {cookies_file}" if os.path.exists(cookies_file) else "",
        "--external-downloader aria2c",
        "--external-downloader-args \"-x 16 -k 1M\""
        ])
        print("Running download command:", download_cmd)
        
        status = os.system(download_cmd)
        if status != 0:
            print(f"Error downloading video: {ytb_id}")
        else:
            print(f"Video {ytb_id} successfully downloaded to {video_path}")
    else:
        print(f"Video already exists: {ytb_id}")

def process_ffmpeg(raw_vid_path, save_folder, save_vid_name, bbox, time):
    """
    Crop video using ffmpeg
    :param raw_vid_path: Path to the source video
    :param save_folder: Save folder
    :param save_vid_name: Name of the saved video
    :param bbox: Crop boundaries (top, bottom, left, right)
    :param time: Start and end in seconds (start_sec, end_sec)
    """
    def secs_to_timestr(secs):
        hrs = secs // 3600
        min = (secs % 3600) // 60
        sec = secs % 60
        return f"{int(hrs):02d}:{int(min):02d}:{int(sec):02d}"

    def expand(bbox, ratio):
        top, bottom = max(bbox[0] - ratio, 0), min(bbox[1] + ratio, 1)
        left, right = max(bbox[2] - ratio, 0), min(bbox[3] + ratio, 1)
        return top, bottom, left, right

    def to_square(bbox):
        top, bottom, left, right = bbox
        h, w = bottom - top, right - left
        side = max(h, w) / 2
        c_y, c_x = (top + bottom) / 2, (left + right) / 2
        return (c_y - side, c_y + side, c_x - side, c_x + side)

    def denorm(bbox, height, width):
        top, bottom = round(bbox[0] * height), round(bbox[1] * height)
        left, right = round(bbox[2] * width), round(bbox[3] * width)
        return top, bottom, left, right

    # Create the final path for saving
    out_path = os.path.join(save_folder, save_vid_name)
    cap = cv2.VideoCapture(raw_vid_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    top, bottom, left, right = to_square(denorm(expand(bbox, 0.02), height, width))

    start_sec, end_sec = time
    # Use system ffmpeg instead of hardcoded path
    ffmpeg_cmd = f"ffmpeg -i {raw_vid_path} -vf crop=w={right-left}:h={bottom-top}:x={left}:y={top} -ss {secs_to_timestr(start_sec)} -to {secs_to_timestr(end_sec)} -loglevel error {out_path}"
    print("Running ffmpeg command")
    status = os.system(ffmpeg_cmd)
    if status != 0:
        print(f"Error processing video: {raw_vid_path}")
    else:
        print(f"Video successfully processed and saved to: {out_path}")

def load_data(file_path):
    """
    Load data from JSON file
    :param file_path: Path to JSON file
    :yield: Fields ytb_id, save_name, time, bbox for each video
    """
    with open(file_path) as f:
        data = json.load(f)
    for key, val in data['clips'].items():
        yield (
            val['ytb_id'],
            f"{key}.mp4",
            (val['duration']['start_sec'], val['duration']['end_sec']),
            [val['bbox']['top'], val['bbox']['bottom'], val['bbox']['left'], val['bbox']['right']]
        )

def get_disk_space(path):
    """
    Get available disk space in GB
    :param path: Path to check
    :return: Available space in GB
    """
    stat = shutil.disk_usage(path)
    return stat.free / (1024 * 1024 * 1024)  # Convert to GB

if __name__ == '__main__':
    # Base directory for data storage

    target_dir = "/mnt/disks"
    
    # Paths for data
    json_path = "celebvhq_info.json"
    raw_vid_root = os.path.join(target_dir, "CelebV-HQ/raw")
    processed_vid_root = os.path.join(target_dir, "CelebV-HQ/processed")
    
    proxy = None  

    # Check available space
    available_space = get_disk_space(target_dir)
    print(f"Available space in {target_dir}: {available_space:.2f} GB")

    # Create directories if they don't exist
    os.makedirs(raw_vid_root, exist_ok=True)
    os.makedirs(processed_vid_root, exist_ok=True)

    for ytb_id, save_name, time, bbox in load_data(json_path):
        raw_path = os.path.join(raw_vid_root, f"{ytb_id}.f137.mp4")
        
        # Download video
        download(raw_path, ytb_id, proxy)
        
        # Process and save
        process_ffmpeg(raw_path, processed_vid_root, save_name, bbox, time)
