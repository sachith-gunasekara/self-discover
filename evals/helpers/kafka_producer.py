import os
import sys
import uuid
import datetime
import json

from kafka import KafkaProducer

from self_discover._helpers.logger import logger


KAFKA_BROKER = "localhost:9092"


def on_send_success(record_metadata):
    logger.info(
        f"[PID {os.getpid()}] Sent OK: Topic={record_metadata.topic} Partition={record_metadata.partition} Offset={record_metadata.offset}"
    )


def on_send_error(excp):
    logger.error(f"[PID {os.getpid()}] Send ERROR: {excp}", file=sys.stderr)


class KafkaProducerHelper:

    def __init__(self, target_topic, producer_key, kafka_broker=KAFKA_BROKER):
        self.kafka_broker = kafka_broker
        self.target_topic = target_topic
        self.producer_key = producer_key

        self.producer = self._initialize_producer()

    def _initialize_producer(self):
        try:
            producer = KafkaProducer(
                bootstrap_servers=self.kafka_broker,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8"),
                acks="all",
            )

            logger.info("Connected producer successfully to {}", self.kafka_broker)
        except Exception as e:
            logger.error("Failed to connect producer: {}", e)
            sys.exit(1)

        return producer

    def generate_event_payload(
        self, process_id, prompt_tokens: int, completion_tokens: int, total_cost: float
    ):
        payload = {
            "eventId": str(uuid.uuid4()),
            "eventTimestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sourceProcessId": process_id,
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": prompt_tokens + completion_tokens,
            "totalCost": total_cost,
        }

        return payload

    def produce(self, payload):
        try:
            future = self.producer.send(
                self.target_topic, value=payload, key=self.producer_key
            )

            future.add_callback(on_send_success)
            future.add_errback(on_send_error)

            self.producer.flush(timeout=10)
            logger.info(f"[PID {payload['sourceProcessId']}] Message flushed")

            return 0
        except Exception as e:
            logger.error(f"[PID {payload['sourceProcessId']}] Failed to send message: {e}")

            return 1
    
    def close(self):
        self.producer.close()

        logger.info("Kafka producer closed.")