class DownloadError(Exception):
    """Error during fetching github assets"""
    pass

class PublishError(Exception):
    """Error during upload of github assets"""
    pass
