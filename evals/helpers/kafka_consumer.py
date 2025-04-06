import os
import sys
import json
import threading

from kafka import KafkaConsumer
from pyprojroot import here

from self_discover._helpers.logger import logger


KAFKA_BROKER = "localhost:9092"
TARGET_TOPIC = "self-discover-cost"
CONSUMER_GROUP_ID = "cost-updater-group"
OUTPUT_FILE = here(f"evals/logs/running_cost.json")


lock = threading.Lock()

if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"running_cost": 0}, f)

def read_running_cost():
    with lock:
        with open(OUTPUT_FILE, "r") as f:
            data = json.load(f)

    return data["running_cost"]

def update_running_cost(cost: float):
    running_cost = read_running_cost() + cost

    with lock:
        with open(OUTPUT_FILE, "w") as f:
            json.dump({"running_cost": running_cost}, f)

def process_message(message_value):
    logger.info(f"[Consumer WORKER] Received message value: {message_value}")

    try:
        update_running_cost(message_value["totalCost"])

        logger.info("[Consumer WORKER] Successfully updated running cost")

        return True
    except Exception as e:
        logger.error("[Consumer WORKER] !!! FAILED to process message and update file: {}", e)
        return False

try:
    consumer = KafkaConsumer(
        TARGET_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        group_id=CONSUMER_GROUP_ID,
        auto_offset_reset='earliest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    )

    logger.info("[Consumer WORKER] Consumer connected. Waiting for messages...")
except Exception as e:
    logger.error("[Consumer WORKER] Failed to connect consumer: {}", e)
    sys.exit(1)

try:
    for message in consumer:
        logger.info("[Consumer WORKER] Consuming: Topic={} Part={} Offset={} Key={}", message.topic, message.partition, message.offset, message.key)

        success = process_message(message.value)
except KeyboardInterrupt:
    logger.warning("[Consumer WORKER] KeyboardInterrupt received. Shutting down...")
except Exception as e:
    logger.error("[Consumer WORKER] An unexpected error occurred: {}", e)
finally:
    consumer.close()
    logger.info("[Consumer WORKER] Consumer closed.")