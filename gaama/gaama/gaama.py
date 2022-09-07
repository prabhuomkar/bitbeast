"""GaAMA"""
from datetime import date
import io
from typing import List
from zipfile import ZipFile
from gaama.github import GitHub


class GaAMA:
    """GaAMA"""
    def __init__(self, username, password, owner, repository) -> None:
        self.github = GitHub(username, password, owner, repository)

    def _get_tag(self) -> str:
        """create release tag"""
        # TODO(omkar): maybe support semVer?
        return str(date.today()).replace('-', '.') # YYYY.MM.DD

    def _create_release(self, tag: str) -> str:
        """create github release"""
        if tag is None:
            tag = self._get_tag()
        res = self.github.create_github_release(tag)
        return res['id']

    def publish(self, tag: str, files: List[str]) -> None:
        """publish model artifacts"""
        release_id = self._create_release(tag)
        zip_file = f'{tag}.zip'
        with ZipFile(zip_file, 'w') as zip_writer:
            for file in files:
                zip_writer.write(file)
        self.github.upload_github_release_assets(release_id, zip_file)

    def download(self, tag: str, path: str = '.') -> None:
        """download model artifacts"""
        artifact_url = self.github.get_github_release_assets(tag)
        res = self.github.download_github_release_assets(artifact_url)
        with ZipFile(io.BytesIO(res.content)) as zip_file:
            zip_file.extractall(path)
