import os
import fcntl


def is_infintetalk_model() -> bool:
    """Check if the model name corresponds to an InfinteTalk model."""
    model_name = os.getenv("MODEL")
    return model_name is not None and "infintetalk" in model_name.lower()


def is_wan_model() -> bool:
    """Check if the model name corresponds to an Wan2.2-TI2V-5B"""
    model_name = os.getenv("MODEL")
    return model_name is not None and "wan" in model_name.lower()


def find_max_matching_frame(max_value: int, default_value: int) -> int:
    """
    Finds the largest integer less than or equal to max_value
    that can be expressed in the form 4*n + 1.

    Args:
        max_value: The upper bound for the search.

    Returns:
        The largest number matching the pattern, or None if no such
        number exists within the given limit (e.g., if max_value < 1).
    """
    # The smallest number of the form 4*n + 1 (for n>=0) is 1.
    if max_value < 1:
        return default_value

    # Start from max_value and check downwards.
    for number in range(max_value, 0, -1):
        # A number is of the form 4*n + 1 if its remainder when divided by 4 is 1.
        if number % 4 == 1:
            return number

    return default_value  # Should not be reached if max_value >= 1


def update_job(job_processed, args):
    # If a job was processed, rewrite the entire job file
    job_file = os.path.join(args.video_dir, "job.txt")
    sep = args.sep
    if job_processed:
        with open(job_file, "r+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                # Re-read the file to get the latest content before writing
                f.seek(0)
                lines_before_write = [line.strip() for line in f if line.strip()]

                # Find the job by ID and update it
                job_id_to_update = job_processed[0]
                found = False
                for i, line in enumerate(lines_before_write):
                    if line.startswith(job_id_to_update + sep):
                        lines_before_write[i] = sep.join(map(str, job_processed))
                        found = True
                        break

                # If the job was somehow removed from the file, add the new status at the end
                if not found:
                    lines_before_write.append(sep.join(map(str, job_processed)))

                # Write the updated content back to the file
                f.seek(0)
                f.truncate()
                for line in lines_before_write:
                    f.write(line + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
