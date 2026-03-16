configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(EAAC_TEST_DEPENDS
    eaac-opt
)

add_lit_testsuite(check-eaac "Running EAAC regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${EAAC_TEST_DEPENDS}
)
