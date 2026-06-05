import pytest
from rag.query_processor import extract_metadata_filters
from rag.retriever import _build_odata_filter

def test_metadata_extraction_beneficiaries():
    query = "what are the schemes for women farmers in punjab?"
    filters = extract_metadata_filters(query)
    
    assert filters.get("state") == "Punjab"
    assert filters.get("is_for_women") is True
    assert filters.get("is_for_farmers") is True

def test_metadata_extraction_students_and_disabled():
    query = "scholarships for disabled students in karnataka"
    filters = extract_metadata_filters(query)
    
    assert filters.get("state") == "Karnataka"
    assert filters.get("is_for_students") is True
    assert filters.get("is_for_disabled") is True

def test_build_odata_filter():
    meta = {
        "state": "Punjab",
        "is_for_women": True,
        "is_for_farmers": True
    }
    
    odata_filter = _build_odata_filter(meta)
    assert "state_or_ut eq 'Punjab'" in odata_filter
    assert "is_for_women eq true" in odata_filter
    assert "is_for_farmers eq true" in odata_filter

def test_build_odata_filter_empty():
    assert _build_odata_filter(None) is None
    assert _build_odata_filter({}) is None
