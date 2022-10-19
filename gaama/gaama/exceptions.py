"""Custom Exceptions"""
class DownloadError(Exception):
    """Error during fetching github assets"""
    pass # pylint: disable=unnecessary-pass

class PublishError(Exception):
    """Error during upload of github assets"""
    pass # pylint: disable=unnecessary-pass
