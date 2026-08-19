# SPDX-License-Identifier: Apache-2.0
FROM python:3.12-slim

WORKDIR /opt/lmcache-qos

COPY lmcache/v1/multiprocess/qos.py /opt/lmcache-qos/qos.py
COPY examples/l2_qos/l2_qos_service.py /opt/lmcache-qos/l2_qos_service.py

ENTRYPOINT ["python", "/opt/lmcache-qos/l2_qos_service.py"]
