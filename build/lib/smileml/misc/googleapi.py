import argparse
import logging
import os
from googleapiclient.discovery import build
import httplib2
from oauth2client import client
from oauth2client import file
from oauth2client import tools
import json
from glob import glob
import pandas as pd

logging.getLogger().setLevel('INFO')


def get_google_service(api_name, api_version, scope, client_secrets_path):
    """Get a service that communicates to a Google API.

    Args
    ----
    `api_name`: string
        The name of the api to connect to (ex: `drive`, `analytics`).
    `api_version`: string
        The api version to connect to (ex: `v3`).
    `scope`: A list of strings
        representing the auth scopes to authorize for the connection.
    `client_secrets_path`: string
        A path to a valid client secrets file. Get one by creating a
        new project here `https://console.cloud.google.com/home/`

    Returns
    -------
    A service that is connected to the specified API.

    Examples
    --------
    For google drive api: `https://developers.google.com/drive/v3/reference/`

    .. code-block:: python

        scope = ['https://www.googleapis.com/auth/drive']
        service = get_google_service('drive', 'v3', scope, '/Users/phi/client_secrets.json')
        service.files().list().execute()  # List files in your drive
    """
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
         formatter_class=argparse.RawDescriptionHelpFormatter,
         parents=[tools.argparser])
    flags = parser.parse_args([])

    # Set up a Flow object to be used if we need to authenticate.
    flow = client.flow_from_clientsecrets(
           client_secrets_path, scope=scope,
           message=tools.message_if_missing(client_secrets_path))
    # Prepare credentials, and authorize HTTP object with them.
    # If the credentials don't exist or are invalid run through the native client
    # flow. The Storage object will ensure that if successful the good
    # credentials will get written back to a file.
    storage = file.Storage(api_name + '.dat')
    credentials = storage.get()
    if credentials is None or credentials.invalid:
        credentials = tools.run_flow(flow, storage, flags)
    http = credentials.authorize(http=httplib2.Http())

    # Build the service object.
    service = build(api_name, api_version, http=http)

    return service


FOLDER = 'application/vnd.google-apps.folder'


def _iterfiles(service, name=None, is_folder=None, parent=None, order_by='folder,name,createdTime'):
    q = []
    if name is not None:
        q.append("name = '%s'" % name.replace("'", "\\'"))
    if is_folder is not None:
        q.append("mimeType %s '%s'" % ('=' if is_folder else '!=', FOLDER))
    if parent is not None:
        q.append("'%s' in parents" % parent.replace("'", "\\'"))
    params = {
        'pageToken': None,
        'orderBy': order_by,
        'fields': 'nextPageToken, files(id, name, mimeType, permissions)'
    }
    if q:
        params['q'] = ' and '.join(q)
    while True:
        response = service.files().list(**params).execute()
        for f in response['files']:
            yield f
        try:
            params['pageToken'] = response['nextPageToken']
        except KeyError:
            return


def walk_google_drive(service, top):
    """ Directory tree generator. Just like `os.walk` but on your google drive.

    Args
    ----
    `service`: The google service
    `top`: The root directory (ex: `LJN`)

    Examples
    --------

    .. code-block:: python

        for path, root, dirs, files in walk_google_drive('LJN'):
            print('%s\t%d %d' % ('/'.join(path) + '/' + root['name'], len(dirs), len(files)))
    """

    top, = _iterfiles(service, name=top, is_folder=True)
    stack = [((top['name'],), [top])]
    while stack:
        path, tops = stack.pop()
        for top in tops:
            dirs, files = is_file = [], []
            for f in _iterfiles(service, parent=top['id']):
                is_file[f['mimeType'] != FOLDER].append(f)
            yield path, top, dirs, files
            if dirs:
                stack.append((path + (top['name'],), dirs))


def _get_analytics_page(service, profile_id, date_from, date_to,
                        dimensions, metrics,
                        page=1, batch=10000):
    """
    Retrieve up to 10000 rows of google analytics results in the `page`.

    Args
    ----
    `service`: The google service
    `profile_id`: The id of the google analytics account
    `date_from`: The start date (ex: `2015-01-01`)
    `date_to`: The end date (ex: `2015-01-01`)
    `dimensions`: The dimensions (ex: [`ga:Date`])
    `metrics`: The metrics (ex: [`ga:pageviews`])
    `page`: The result page
    `batch`: The number of records to retrieve by request (should not greater than 10000)
    """

    return service.data().ga().get(
          ids='ga:%s' % str(profile_id),
          start_date=date_from,
          end_date=date_to,
          dimensions=','.join(dimensions),
          metrics=','.join(metrics),
          start_index=page,
          max_results=batch).execute()


def get_analytics_results(service, profile_id, date_from, date_to,
                          dimensions, metrics, cacheDir='./temp', batch=10000, exception_if_sampled=True):
    """
    Retrieve all the analytics in the date range.

    Args
    ----
    `service`: The google service
    `profile_id`: The id of the google analytics account
    `date_from`: The start date (ex: `2015-01-01`)
    `date_to`: The end date (ex: `2015-01-01`)
    `dimensions`: The dimensions (ex: [`ga:Date`])
    `metrics`: The metrics (ex: [`ga:pageviews`])
    `cacheDir`: The directory to save intermediate results (for stop and resume purpose)
    `batch`: The number of records to retrieve by request (should not greater than 10000)
    `exception_if_sampled`: If there are to many results, google may decide to sample them.
        This flag is to whether an exception should be throw

    Returns
    -------
    A pandas dataframe
    """
    assert batch <= 10000
    if not os.path.isdir(cacheDir):
        os.mkdir(cacheDir)

    # Get or load first result page
    idx_start = 1
    filename = '%s/result_from_%s_to_%s_%i.json' % (cacheDir, date_from, date_to, idx_start)
    if not os.path.isfile(filename):
        results = _get_analytics_page(service, profile_id, date_from, date_to, dimensions, metrics, idx_start)
        with open(filename, 'w') as fout:
            json.dump(results, fout, indent=4)
    else:
        results = json.load(open(filename, 'r'))

    # Warn if containsSampledData
    if results.get('containsSampledData', False):
        if not exception_if_sampled:
            print('WARNING: The result contains sampled data.' +
                  'If you want to retrieve all the data, try reduce the date range.')
        else:
            raise Exception('The result contains sampled data. ' +
                            'Please reduce the date range or set exception_if_sampled to False')

    # Get the next pages
    counter = len(results.get('rows', []))
    while results["totalResults"] > counter:
        idx_start = counter + 1
        filename = '%s/result_from_%s_to_%s_%i.json' % (cacheDir, date_from, date_to, idx_start)
        if not os.path.isfile(filename):
            results = _get_analytics_page(service, profile_id, date_from, date_to, dimensions, metrics, idx_start)
            with open(filename, 'w') as fout:
                json.dump(results, fout, indent=4)
        else:
            results = json.load(open(filename, 'r'))
        counter += len(results.get('rows', []))

    # Merge the outputs
    files = sorted(glob(cacheDir + '/*.json'))
    records = []
    for fname in files:
        with open(fname, 'r') as fjson:
            data = json.load(fjson)
            header = [rec["name"] for rec in data["columnHeaders"]]
            records += data.get('rows', [])
    return pd.DataFrame(records, columns=header)
