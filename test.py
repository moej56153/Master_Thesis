from pyspi.utils.data_builder.time_series_builder import TimeSeriesBuilderSPI
import numpy as np
from pyspi.utils.function_utils import find_response_version
from pyspi.utils.response.spi_response_data import ResponseDataRMF
from pyspi.utils.response.spi_response import ResponseRMFGenerator
from pyspi.utils.response.spi_drm import SPIDRM
from pyspi.utils.livedets import get_live_dets

grb_time = "120711 024453"
ebounds = np.geomspace(20,520,40)
ein = np.geomspace(20,800,150)
t1 = 3500
t2 = 3800
version = find_response_version(grb_time)
rsp_base = ResponseDataRMF.from_version(version)
dets = get_live_dets(time=grb_time, event_types=["single"])