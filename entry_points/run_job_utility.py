from utility.job_utility import job_creation
class RunJobUtility:

    @staticmethod
    def run_make_jobs_dirs(root: str, instances: int, output_dir: str, abs_paths: bool):
        job_creation.make_jobs_dirs(root, instances, output_dir, abs_paths)
        return 0

    @staticmethod
    def run_make_jobs_ext(root: str, instances: int, extensions: [str], output_dir: str, abs_paths: bool, recursive: bool):
        job_creation.make_jobs_ext(root, instances, extensions, output_dir, abs_paths, recursive)
        return 0

    @staticmethod
    def run_make_jobs_all(root: str, instances: int, output_dir: str, abs_paths: bool, recursive: bool):
        job_creation.make_jobs_all(root, instances, output_dir, abs_paths, recursive)
        return 0


def main():
    pass

if __name__ == "__main__":
    main()