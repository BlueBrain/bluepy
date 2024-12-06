"""Utilities for URL interpretation."""

try:
    import urlparse
except ImportError:
    # pylint: disable=no-name-in-module,import-error
    from urllib import parse as urlparse


def is_local_path(url):
    """true if path doesn't need to be looked up by entity management"""
    scheme = urlparse.urlsplit(url).scheme
    return scheme in ('file', '')


def get_file_path_by_url(url):
    """ Get locally available file path from URL. """
    scheme, netloc, path = urlparse.urlsplit(url)[:3]
    if scheme in ('file', ''):
        assert not netloc
        result = path
    else:
        raise NotImplementedError("Entity management has been removed.")
    return result
