"""GaAMA"""
from datetime import date
import io
import logging
import os
from typing import List
from zipfile import ZipFile

from gaama.github import GitHub


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GaAMA:
    """GaAMA"""
    def __init__(self, username, password, owner, repository) -> None:
        self.github = GitHub(username, password, owner, repository)

    def _get_tag(self) -> str:
        """create release tag"""
        # TODO(omkar): maybe support semVer?
        return str(date.today()).replace('-', '.') # YYYY.MM.DD

    def _create_release(self, tag: str, **kwargs) -> str:
        """create github release"""
        if tag is None:
            tag = self._get_tag()
        res = self.github.create_github_release(tag, **kwargs)
        return res['id']

    def publish(self, tag: str, files: List[str], zip_files: bool = True, **kwargs) -> None:
        """publish model artifacts"""
        logger.info('publishing total %d github assets with tag: %s, zip: %r', len(files), tag, zip_files)
        # get github release id
        release_id = self._create_release(tag, **kwargs)
        # if zip is enabled, zip all the artifacts
        if zip_files:
            zip_file = f'{tag}.zip'
            with ZipFile(zip_file, 'w') as zip_writer:
                for file in files:
                    zip_writer.write(file)
            # upload the zip to github
            self.github.upload_github_release_assets(release_id, zip_file)
            # delete the zip file after upload is finished
            os.remove(zip_file)
        # upload all the files to github
        for file in files:
            self.github.upload_github_release_assets(release_id, file)

    def download(self, tag: str, path: str = '.') -> None:
        """download model artifacts"""
        logger.info('downloading github assets with tag: %s', tag)
        # get list of file name and file download url
        artifacts = self.github.get_github_release_assets(tag)
        for artifact in artifacts:
            # download each artifact
            res = self.github.download_github_release_assets(artifact['url'])
            # unzip if the file contains a zip extension
            if '.zip' in artifact['name']:
                with ZipFile(io.BytesIO(res.content)) as zip_file:
                    zip_file.extractall(path)
            else:
                # normal download the file
                res.raise_for_status()
                with open(artifact['name'], 'wb') as writer:
                    for chunk in res.iter_content(chunk_size=1024):
                        writer.write(chunk)
