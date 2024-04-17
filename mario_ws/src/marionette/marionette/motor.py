#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import lgpio as sbc

class ChipLineNode(Node):
    def __init__(self):
        super().__init__('chip_line_node')
        self.get_logger().info('Initializing Chip Line Node...')

        h = sbc.gpiochip_open(0)
        ci = sbc.gpio_get_chip_info(h)
        self.get_logger().info(f"lines={ci[1]} name={ci[2]} label={ci[3]}")

        for i in range(ci[1]):
            li = sbc.gpio_get_line_info(h, i)
            self.get_logger().info(f"offset={li[1]} flags={li[2]} name={li[3]} user={li[4]}")

        sbc.gpiochip_close(h)

def main(args=None):
    rclpy.init(args=args)
    node = ChipLineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

