package neural

type Layer struct {
	ActivationFunc string `json:"activation-func"`
	Neurons        int    `json:"neurons"`
}
