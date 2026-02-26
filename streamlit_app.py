#!/usr/bin/env python3
"""
Default Streamlit entrypoint.
Delegates to production app that requires BOTH:
- GSC CSV
- Strategic priority keywords file (Excel/CSV)
"""

from streamlit_app_production import main


if __name__ == '__main__':
    main()
