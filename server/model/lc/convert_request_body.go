package lc

type Metadata struct {
	Source                 string  `json:"_source"`
	Position               int     `json:"position"`
	DatasetID              string  `json:"dataset_id"`
	DatasetName            string  `json:"dataset_name"`
	DocumentID             string  `json:"document_id"`
	DocumentName           string  `json:"document_name"`
	DocumentDataSourceType string  `json:"document_data_source_type"`
	SegmentID              string  `json:"segment_id"`
	RetrieverFrom          string  `json:"retriever_from"`
	Score                  float64 `json:"score"`
	SegmentHitCount        int     `json:"segment_hit_count"`
	SegmentWordCount       int     `json:"segment_word_count"`
	SegmentPosition        int     `json:"segment_position"`
	SegmentIndexNodeHash   string  `json:"segment_index_node_hash"`
}

type Result struct {
	Metadata Metadata `json:"metadata"`
	Title    string   `json:"title"`
	Content  string   `json:"content"`
}

type ConvertRequest struct {
	Result []Result `json:"result"`
}
