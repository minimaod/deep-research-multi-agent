"""Only expose project skill assets; leave other agent files in StateBackend."""
from pathlib import Path
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol, EditResult, FileDownloadResponse, FileUploadResponse, WriteResult


class ReadOnlySkillBackend(BackendProtocol):
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.delegate = FilesystemBackend(root_dir=self.root, virtual_mode=True)

    def ls(self, path):
        return self.delegate.ls(path)

    def read(self, file_path, offset=0, limit=2000):
        return self.delegate.read(file_path, offset, limit)

    def grep(self, pattern, path=None, glob=None):
        return self.delegate.grep(pattern, path, glob)

    def glob(self, pattern, path="/"):
        return self.delegate.glob(pattern, path)

    def download_files(self, paths):
        replies = []
        for path in paths:
            target = (self.root / path.lstrip("/")).resolve()
            if not target.is_relative_to(self.root) or ".." in Path(path).parts:
                replies.append(FileDownloadResponse(path=path, error="permission_denied"))
            else:
                replies.extend(self.delegate.download_files([path]))
        return replies

    def write(self, file_path, content):
        return WriteResult(error="skills_are_read_only")

    def edit(self, file_path, old_string, new_string, replace_all=False):
        return EditResult(error="skills_are_read_only")

    def upload_files(self, files):
        return [FileUploadResponse(path=path, error="permission_denied") for path, _ in files]


def make_skill_backend(_runtime=None):
    root = Path(__file__).resolve().parents[2] / "skills"
    return CompositeBackend(default=StateBackend(), routes={"/skills/": ReadOnlySkillBackend(root)})
