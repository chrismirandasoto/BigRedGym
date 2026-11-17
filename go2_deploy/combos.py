"""Remote button combos for go2_deploy.launcher (torch- and SDK-free).

Bits of wireless_remote byte 2, as in UnitreeRemoteController.parse_button.
"""

START = 0b000011  # L1 + R1
STOP = 0b110000  # L2 + R2


def held(wireless_remote, combo):
    return wireless_remote[2] & combo == combo


class ComboDetector:
    """Fires once per press, after the combo has been held for hold_s seconds."""

    def __init__(self, hold_s=1.0):
        self.hold_s = hold_s
        self._since = None
        self._fired = False

    def update(self, pressed, now):
        if not pressed:
            self._since, self._fired = None, False
            return False
        if self._since is None:
            self._since = now
        if self._fired or now - self._since < self.hold_s:
            return False
        self._fired = True
        return True
