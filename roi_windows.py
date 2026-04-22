from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Canonical ROI mapping requested by user
ROI_DISTANCES_CM: List[int] = [25, 50, 75, 100, 125]
ROI_STARTS: List[int] = [2915, 5800, 8650, 11550, 14400]
ROI_LEN: int = 630

# Distance -> (start, end) using inclusive start / exclusive end indexing
ROI_WINDOWS: Dict[int, Tuple[int, int]] = {
    d: (s, s + ROI_LEN) for d, s in zip(ROI_DISTANCES_CM, ROI_STARTS)
}


def parse_distance_cm(filename: str) -> Optional[int]:
    """Extract distance in cm from a filename like '..._25cm_...'."""
    m = re.search(r"(\d+)\s*cm", str(filename))
    return int(m.group(1)) if m else None
